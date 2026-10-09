"""Render a multi-turn conversation with the model's chat template and locate token spans.

For Gemma 3 the template is
    <bos><start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>\n ...
Gemma 4 uses the same start/end-of-turn tokens (verified on the pod by `check_template`).

`render()` returns input_ids plus, for every turn, the content span [start, end) and for every
assistant turn the *response-preparation* index: the last template token before the assistant's
content (the "\n" after "model"), i.e. Anthropic's "colon after Assistant".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Rendered:
    input_ids: torch.Tensor                 # [T]
    text: str
    turns: list[dict] = field(default_factory=list)   # {"role", "start", "end", "prep": int|None, "turn_idx": k}

    def assistant_turns(self):
        return [t for t in self.turns if t["role"] == "assistant"]


def _find_token(tok, s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{s!r} is not a single token in this tokenizer: {ids}")
    return ids[0]


def role_tokens(tok) -> tuple[int, int]:
    """(start_of_turn_id, end_of_turn_id). Gemma 3: <start_of_turn>/<end_of_turn>; Gemma 4: <|turn>/<turn|>."""
    sot_s = getattr(tok, "sot_token", None) or "<start_of_turn>"
    eot_s = getattr(tok, "eot_token", None) or "<end_of_turn>"
    return _find_token(tok, sot_s), _find_token(tok, eot_s)


def _is_gemma4(tok) -> bool:
    return getattr(tok, "sot_token", None) == "<|turn>"


_G4_MODEL_HEADER = "<|turn>model\n"
_G4_EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


def _is_gemma(tok) -> bool:
    return _is_gemma4(tok) or "<start_of_turn>" in tok.get_vocab()


def _render_generic(tok, messages: list[dict], add_generation_prompt: bool = False) -> Rendered:
    """Any chat template (Qwen, OLMo, gpt-oss, ...): render, then find each message's content by character position
    and map it to tokens with the fast tokenizer's offsets. The response-prep index is the last token before an
    assistant turn's content; with add_generation_prompt the open assistant turn starts at the end of the text.
    System messages get role 'system' (templates such as gpt-oss move them into a developer block)."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   **getattr(tok, "dprobe_chat_kwargs", {}))
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor(enc["input_ids"])
    starts = torch.tensor([a for a, _ in enc["offset_mapping"]])
    ends = torch.tensor([b for _, b in enc["offset_mapping"]])
    turns, cursor, n = [], 0, {"user": 0, "assistant": 0}
    for m in messages:
        content = m["content"] or ""
        if not content.strip():
            continue
        i, piece = text.find(content, cursor), content
        if i < 0:                                   # templates may strip surrounding whitespace
            piece = content.strip()
            i = text.find(piece, cursor)
        if i < 0:
            raise ValueError(f"cannot locate a {m['role']} message in the rendered template")
        cs, ce = i, i + len(piece)
        cursor = ce
        start = int((ends > cs).nonzero()[0])
        after = (starts >= ce).nonzero()
        end = int(after[0]) if len(after) else len(ids)
        role = m["role"] if m["role"] in ("user", "assistant") else "system"
        k = n.get(role, 0)
        turns.append({"role": role, "start": start, "end": end, "prep": start - 1 if role == "assistant" else None, "turn_idx": k})
        if role in n:
            n[role] += 1
    if add_generation_prompt:
        turns.append({"role": "assistant", "start": len(ids), "end": len(ids), "prep": len(ids) - 1, "turn_idx": n["assistant"]})
    return Rendered(ids, text, turns)


def render(tok, messages: list[dict], add_generation_prompt: bool = False, match_generation: bool = True) -> Rendered:
    """Render with the chat template and locate spans (non-Gemma templates: see _render_generic).

    match_generation (Gemma 4 only): the canonical template emits an empty thought block
    `<|channel>thought\n<channel|>` after `<|turn>model\n` for the *generation prompt* but not for
    already-written assistant turns. At generation time (thinking off) the model saw that block, so
    for teacher-forced probing we insert it before every assistant turn; the response-prep token is
    then `<channel|>`, the last template token before the assistant's content.
    """
    if not _is_gemma(tok):
        return _render_generic(tok, messages, add_generation_prompt)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    if match_generation and _is_gemma4(tok):
        text = text.replace(_G4_MODEL_HEADER + _G4_EMPTY_THOUGHT, "\x00").replace(_G4_MODEL_HEADER, _G4_MODEL_HEADER + _G4_EMPTY_THOUGHT).replace("\x00", _G4_MODEL_HEADER + _G4_EMPTY_THOUGHT)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    sot, eot = role_tokens(tok)
    id_list = ids.tolist()
    turns = []
    i = 0
    n_user = n_asst = 0
    while i < len(id_list):
        if id_list[i] != sot:
            i += 1
            continue
        # role name is the next token(s) up to and including the newline token
        j = i + 1
        role_piece = ""
        while j < len(id_list) and "\n" not in role_piece:
            role_piece += tok.decode([id_list[j]])
            j += 1
        role = "assistant" if role_piece.strip().startswith("model") else ("user" if role_piece.strip().startswith("user") else role_piece.strip())
        # Gemma 4: skip an empty thought block "<|channel>thought\n<channel|>" so content starts after it
        soc = getattr(tok, "soc_token", None)
        eoc = getattr(tok, "eoc_token", None)
        if soc and eoc and j < len(id_list) and tok.decode([id_list[j]]) == soc:
            jj = j
            while jj < len(id_list) and tok.decode([id_list[jj]]) != eoc:
                jj += 1
            if jj < len(id_list) and jj - j <= 4:
                j = jj + 1
        prep = j - 1                       # last template token before content ("\n" for Gemma 3, "<channel|>" for Gemma 4)
        start = j
        end = start
        while end < len(id_list) and id_list[end] != eot:
            end += 1
        k = n_asst if role == "assistant" else n_user
        turns.append({"role": role, "start": start, "end": end, "prep": prep if role == "assistant" else None, "turn_idx": k})
        if role == "assistant":
            n_asst += 1
        else:
            n_user += 1
        i = end + 1
    return Rendered(ids, text, turns)


def check_template(tok, verbose: bool = True) -> bool:
    msgs = [{"role": "user", "content": "Hi there"}, {"role": "assistant", "content": "Hello! How can I help?"}, {"role": "user", "content": "Bye"}]
    r = render(tok, msgs, add_generation_prompt=True)
    ok = True
    expected = {("user", 0): msgs[0]["content"], ("assistant", 0): msgs[1]["content"], ("user", 1): msgs[2]["content"], ("assistant", 1): ""}
    for t in r.turns:
        piece = tok.decode(r.input_ids[t["start"]:t["end"]])
        exp = expected.get((t["role"], t["turn_idx"]), "")
        if piece.strip() != exp.strip():
            ok = False
        if verbose:
            prep = tok.decode(r.input_ids[t["prep"]]) if t["prep"] is not None else None
            print(f"  {t['role']:9s} turn {t['turn_idx']}: [{t['start']},{t['end']}) -> {piece!r}  prep={prep!r}")
    if verbose:
        print(f"  rendered: {r.text!r}")
        print(f"  template check: {'OK' if ok else 'MISMATCH'}")
    return ok

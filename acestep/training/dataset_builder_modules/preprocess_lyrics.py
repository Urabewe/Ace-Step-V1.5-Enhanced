import torch


def encode_lyrics(text_encoder, text_tokenizer, lyrics: str, device, dtype):
    """Encode lyrics into hidden states."""
    lyric_inputs = text_tokenizer(
        lyrics,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    lyric_input_ids = lyric_inputs.input_ids.to(device)
    lyric_attention_mask = lyric_inputs.attention_mask.to(device).to(dtype)

    enc_device = next(text_encoder.parameters()).device
    if lyric_input_ids.device != enc_device:
        lyric_input_ids = lyric_input_ids.to(enc_device)
        lyric_attention_mask = lyric_attention_mask.to(enc_device)

    with torch.no_grad():
        lyric_hidden_states = text_encoder.embed_tokens(lyric_input_ids).to(dtype)

    return lyric_hidden_states, lyric_attention_mask

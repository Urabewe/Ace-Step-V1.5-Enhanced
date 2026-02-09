import torch


def run_encoder(
    model,
    text_hidden_states,
    text_attention_mask,
    lyric_hidden_states,
    lyric_attention_mask,
    device,
    dtype,
):
    """Run model encoder to get hidden states and attention mask."""
    try:
        encoder_device = next(model.encoder.parameters()).device
    except Exception:
        encoder_device = device

    if text_hidden_states.device != encoder_device:
        text_hidden_states = text_hidden_states.to(encoder_device)
    if text_attention_mask.device != encoder_device:
        text_attention_mask = text_attention_mask.to(encoder_device)
    if lyric_hidden_states.device != encoder_device:
        lyric_hidden_states = lyric_hidden_states.to(encoder_device)
    if lyric_attention_mask.device != encoder_device:
        lyric_attention_mask = lyric_attention_mask.to(encoder_device)

    refer_audio_hidden = torch.zeros(1, 1, 64, device=encoder_device, dtype=dtype)
    refer_audio_order_mask = torch.zeros(1, device=encoder_device, dtype=torch.long)

    with torch.no_grad():
        encoder_hidden_states, encoder_attention_mask = model.encoder(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_hidden,
            refer_audio_order_mask=refer_audio_order_mask,
        )

    return encoder_hidden_states, encoder_attention_mask

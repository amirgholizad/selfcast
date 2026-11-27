import torch

from selfcast.engine import S3GEN_BUCKET, _s3gen_bucketed


def test_s3gen_input_is_padded_to_a_bucket_and_output_trimmed_back():
    seen = []

    def fake_inference(tokens, ref_dict=None, n_cfm_timesteps=2):
        seen.append(tokens.clone())
        return torch.ones(1, tokens.size(-1) * 960), None   # 960 samples per token, like S3Gen

    wrapped = _s3gen_bucketed(fake_inference, silence_token=4299)
    tokens = torch.arange(73)
    wav, _ = wrapped(tokens, ref_dict={}, n_cfm_timesteps=2)
    assert seen[0].size(-1) % S3GEN_BUCKET == 0             # one of a few fixed lengths
    assert torch.equal(seen[0][:73], tokens)                 # the real tokens are untouched
    assert (seen[0][73:] == 4299).all()                      # padded with silence
    assert wav.size(-1) == 73 * 960                          # and the padding is cut off again

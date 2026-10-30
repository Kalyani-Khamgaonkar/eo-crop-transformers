"""Patch core/models/builder.py and the protocol for groups / concat / conv. Safe to run twice."""
import re
from pathlib import Path

p = Path("core/models/builder.py"); s = p.read_text()
if "pe_mode" not in s:
    rep = [
        ('        self.tokenizer = build_tokenizer(m["tokenizer"], d["bands"], dim, p, pt)\n',
         '        self.pe_mode = m.get("pe_mode", "add")                      # add | concat | conv\n'
         '        self.G = 3 if m["tokenizer"] == "groups" else 1\n'
         '        if m["tokenizer"] == "groups":\n'
         '            from core.models.pe_variants import SpectralGroupTokenizer\n'
         '            assert pt == 1, "groups uses one date per token: set tubelet_t to 1"\n'
         '            self.tokenizer = SpectralGroupTokenizer(d["bands"], dim, p)\n'
         '        else:\n'
         '            self.tokenizer = build_tokenizer(m["tokenizer"], d["bands"], dim, p, pt)\n'),
        ('        self.temporal_enc = TemporalEncoding(m["temporal_enc"], dim, n_tokens_t)\n',
         '        self.temporal_enc = TemporalEncoding(m["temporal_enc"], dim, n_tokens_t)\n'
         '        self.group_enc = m.get("group_enc", "learned")              # learned | sin (groups only)\n'
         '        use_eg = self.G > 1 and not (self.pe_mode == "concat" and self.group_enc == "sin")\n'
         '        self.group_emb = nn.Parameter(torch.zeros(self.G, dim)) if use_eg else None\n'
         '        if self.group_emb is not None:\n'
         '            nn.init.trunc_normal_(self.group_emb, std=0.02)\n'
         '        if self.pe_mode != "add":\n'
         '            assert m["spatial_pe"] == "none", "pe_mode concat/conv: set spatial_pe none"\n'
         '            assert m["temporal_enc"] == "doy_sin", "pe_mode concat/conv: set temporal_enc doy_sin (not used, no parameters)"\n'
         '        if self.G > 1 and self.rel_bias is not None:\n'
         '            raise ValueError("relative spatial PE is not supported with spectral groups")\n'),
        ('        self.head = PixelHead(dim, d["num_classes"], p, self.gh, self.gw)\n',
         '        if self.pe_mode == "conv":\n'
         '            assert self.attention == "joint", "conv PE is implemented for joint attention"\n'
         '            from core.models.pe_variants import ConvMLP\n'
         '            for blk in self.encoder.blocks:\n'
         '                blk.mlp = ConvMLP(dim, m["mlp_ratio"], m["dropout"])\n'
         '        self.head = PixelHead(dim, d["num_classes"], p, self.gh, self.gw)\n'),
    ]
    for old, new in rep:
        assert s.count(old) == 1, "anchor not found: " + old.strip()[:60]
        s = s.replace(old, new)
    a = s.index("        tok, doy_t, mask_t = self.tokenizer(img, doy, mask)")
    e_marker = "            x = masked_mean(x, mask_t[:, :, None].expand(b, t, n), dim=1)   # (B, N, D)\n"
    assert s.count(e_marker) == 1, "anchor not found: joint masked_mean"
    e = s.index(e_marker) + len(e_marker)
    s = s[:a] + '''        tok, doy_t, mask_t = self.tokenizer(img, doy, mask)          # (B, T', G*N, D)
        b, t, gn, dim = tok.shape
        G = self.G
        n = gn // G
        if self.pe_mode == "add":
            spe = self.spatial_pe()
            if spe is not None:
                tok = tok + spe.repeat(G, 1)[None, None]
            tok = tok + self.temporal_enc(doy_t)[:, :, None, :]
        elif self.pe_mode == "concat":
            from core.models.pe_variants import concat_sincos_pe
            tok = tok + concat_sincos_pe(doy_t, self.gh, self.gw, G, dim, group_sin=(G > 1 and self.group_enc == "sin"))
        if self.group_emb is not None:
            tok = tok + self.group_emb.repeat_interleave(n, dim=0)[None, None]
        bias = self.rel_bias() if self.rel_bias is not None else None  # (h, N, N)

        if self.attention == "joint":
            x = tok.reshape(b, t * gn, dim)                           # time-major, then group, then grid
            key_mask = mask_t[:, :, None].expand(b, t, gn).reshape(b, t * gn)
            full_bias = bias.repeat(1, t, t) if bias is not None else None
            if self.pe_mode == "conv":
                for blk in self.encoder.blocks:
                    blk.mlp.set_grid(t, G, self.gh, self.gw, mask_t)
            x = self.encoder(x, key_mask, full_bias).view(b, t, gn, dim)
            x = masked_mean(x, mask_t[:, :, None].expand(b, t, gn), dim=1)   # (B, G*N, D)
            if G > 1:
                x = x.view(b, G, n, dim).mean(1)                       # merge spectral groups before the head
''' + s[e:]
    s += '''

_tokens_per_sample_base = tokens_per_sample


def tokens_per_sample(cfg: dict) -> int:  # noqa: F811  (spectral groups: 3 tokens per date and patch)
    m = cfg["model"]
    if m["tokenizer"] == "groups":
        return 3 * _tokens_per_sample_base({**cfg, "model": {**m, "tokenizer": "patch2d", "tubelet_t": 1}})
    return _tokens_per_sample_base(cfg)
'''
    p.write_text(s); print("builder.py patched")
else:
    print("builder.py already patched")

pp = Path("experiments/protocol.yaml") if Path("experiments/protocol.yaml").exists() else [x for x in Path(".").rglob("protocol.yaml") if ".git" not in x.parts][0]
s = pp.read_text()
if "pe_mode" not in s:
    m = re.search(r"^  temporal_enc: .*\n", s, re.M); assert m, "temporal_enc line not found in protocol"
    s = s.replace(m.group(0), m.group(0) +
                  "  pe_mode: add                 # add | concat | conv   (how position enters the token)\n"
                  "  group_enc: learned           # learned | sin         (spectral groups only)\n")
    s = s.replace("  - model.temporal_enc\n", "  - model.temporal_enc\n  - model.pe_mode\n  - model.group_enc\n")
    s = re.sub(r"^(  tokenizer: patch2d\s+# patch2d \| tubelet \| cuboid)", r"\1 | groups", s, flags=re.M)
    pp.write_text(s); print(pp, "patched")
else:
    print(pp, "already patched")

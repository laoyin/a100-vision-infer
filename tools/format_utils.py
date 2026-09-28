"""AVI v1 numeric layout and TP partition rules; NumPy only."""
import numpy as np


def bf16_bytes(values):
    values = np.asarray(values, dtype='<f4').copy()
    bits = values.view('<u4')
    rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & 1)
    return (rounded >> 16).astype('<u2').tobytes()


def decode_fp8(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    e = ((codes >> 3) & 15).astype(np.int32)
    m = (codes & 7).astype(np.float32)
    value = np.where(e == 0, np.ldexp(m, -9), np.ldexp(1 + m / 8, e - 7))
    value = np.where((e == 15) & (m == 7), np.nan, value)
    return np.where(codes & 128, -value, value).astype(np.float32)


_FP8_POSITIVE = decode_fp8(np.arange(127, dtype=np.uint8))


def quantize_fp8(values):
    """E4M3FN, per-output-row FP32 multiplier, ties to even code."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('FP8 input must be a finite matrix')
    scales = np.maximum(np.max(np.abs(values), axis=1) / 448, np.finfo(np.float32).tiny)
    codes = np.empty(values.shape, dtype=np.uint8)
    for start in range(0, len(values), 256):
        batch = values[start:start + 256]
        target = np.minimum(np.abs(batch) / scales[start:start + 256, None], 448)
        upper = np.searchsorted(_FP8_POSITIVE, target).clip(0, 126)
        lower = np.maximum(upper - 1, 0)
        dl, du = target - _FP8_POSITIVE[lower], _FP8_POSITIVE[upper] - target
        use_upper = (du < dl) | ((du == dl) & ((upper & 1) == 0))
        selected = np.where(use_upper, upper, lower).astype(np.uint8)
        codes[start:start + len(batch)] = selected | (np.signbit(batch).astype(np.uint8) << 7)
    return codes, scales.astype('<f4')


def partition(name, values, text, rank, tp):
    if tp not in (1, 2, 4) or not 0 <= rank < tp:
        raise ValueError('Invalid rank/TP')
    if tp == 1 or not name.startswith('model.language_model.layers.'):
        return values
    def chunk(x, axis=0):
        if x.shape[axis] % tp:
            raise ValueError(f'{name}: dimension not divisible by TP')
        return np.split(x, tp, axis=axis)[rank]
    if name.endswith(('.mlp.gate_proj.weight', '.mlp.up_proj.weight',
                      '.self_attn.q_proj.weight', '.self_attn.k_proj.weight', '.self_attn.v_proj.weight',
                      '.linear_attn.in_proj_z.weight', '.linear_attn.in_proj_a.weight', '.linear_attn.in_proj_b.weight',
                      '.linear_attn.A_log', '.linear_attn.dt_bias')):
        return chunk(values)
    if name.endswith(('.mlp.down_proj.weight', '.self_attn.o_proj.weight', '.linear_attn.out_proj.weight')):
        return chunk(values, 1)
    if name.endswith(('.linear_attn.in_proj_qkv.weight', '.linear_attn.conv1d.weight')):
        key = text['linear_num_key_heads'] * text['linear_key_head_dim']
        value = text['linear_num_value_heads'] * text['linear_value_head_dim']
        if values.shape[0] != key * 2 + value:
            raise ValueError(f'{name}: unexpected fused QKV shape')
        q, k, v = np.split(values, [key, key * 2], axis=0)
        return np.concatenate([chunk(q), chunk(k), chunk(v)], axis=0)
    return values


def image_geometry(height, width, merge=2, side=48):
    if height % merge or width % merge or height <= 0 or width <= 0:
        raise ValueError('Invalid image grid')
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    order = np.arange(height * width).reshape(height // merge, merge, width // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
    coords = np.stack([rows.reshape(-1)[order], cols.reshape(-1)[order]], axis=-1).astype('<i8')
    h = np.linspace(0, side - 1, height, dtype=np.float32)
    w = np.linspace(0, side - 1, width, dtype=np.float32)
    hf, wf = h.astype(int), w.astype(int)
    hc, wc = np.minimum(hf + 1, side - 1), np.minimum(wf + 1, side - 1)
    dh, dw = h - hf, w - wf
    indices = np.stack([(r[:, None] * side + c[None, :]).reshape(-1)[order]
                        for r, c in [(hf, wf), (hf, wc), (hc, wf), (hc, wc)]]).astype('<i8')
    factors = np.stack([(r[:, None] * c[None, :]).reshape(-1)[order]
                        for r, c in [(1-dh, 1-dw), (1-dh, dw), (dh, 1-dw), (dh, dw)]]).astype('<f4')
    return coords, indices, factors


def rope_positions(ids, image_grids, image_token, merge):
    ids = list(ids)
    positions = np.zeros((3, len(ids)), dtype='<i8')
    i = 0
    current = 0
    grids = iter(image_grids)
    used = 0
    while i < len(ids):
        if ids[i] != image_token:
            positions[:, i] = current
            current += 1
            i += 1
            continue
        try:
            t, h, w = next(grids)
        except StopIteration as exc:
            raise ValueError('Image tokens without grid') from exc
        if t != 1 or h % merge or w % merge:
            raise ValueError('v0.1 supports still images with divisible grids')
        h, w = h // merge, w // merge
        n = h * w
        if ids[i:i+n] != [image_token] * n:
            raise ValueError('Image token count does not match grid')
        positions[0, i:i+n] = current
        positions[1, i:i+n] = np.repeat(np.arange(h), w) + current
        positions[2, i:i+n] = np.tile(np.arange(w), h) + current
        current += max(h, w)
        i += n
        used += 1
    if used != len(image_grids):
        raise ValueError('Unused image grid')
    return positions, int(positions.max()) + 1

def expected_shapes(config):
    t, v = config['text_config'], config['vision_config']
    h, mid, d = t['hidden_size'], t['intermediate_size'], t['head_dim']
    q, kv = t['num_attention_heads']*d, t['num_key_value_heads']*d
    kd = t['linear_num_key_heads']*t['linear_key_head_dim']
    vd = t['linear_num_value_heads']*t['linear_value_head_dim']
    result={'lm_head.weight':(t['vocab_size'],h),
            'model.language_model.embed_tokens.weight':(t['vocab_size'],h),
            'model.language_model.norm.weight':(h,)}
    def linear(prefix, out, inp, bias=False):
        result[prefix+'.weight']=(out,inp)
        if bias: result[prefix+'.bias']=(out,)
    for i, kind in enumerate(t['layer_types']):
        p=f'model.language_model.layers.{i}'
        result[p+'.input_layernorm.weight']=(h,)
        result[p+'.post_attention_layernorm.weight']=(h,)
        linear(p+'.mlp.gate_proj',mid,h); linear(p+'.mlp.up_proj',mid,h); linear(p+'.mlp.down_proj',h,mid)
        if kind=='full_attention':
            a=p+'.self_attn'
            linear(a+'.q_proj',q*2,h); linear(a+'.k_proj',kv,h); linear(a+'.v_proj',kv,h); linear(a+'.o_proj',h,q)
            result[a+'.q_norm.weight']=(d,); result[a+'.k_norm.weight']=(d,)
        else:
            a=p+'.linear_attn'
            linear(a+'.in_proj_qkv',2*kd+vd,h); linear(a+'.in_proj_z',vd,h)
            linear(a+'.in_proj_a',t['linear_num_value_heads'],h); linear(a+'.in_proj_b',t['linear_num_value_heads'],h)
            linear(a+'.out_proj',h,vd)
            result[a+'.conv1d.weight']=(2*kd+vd,1,t['linear_conv_kernel_dim'])
            result[a+'.A_log']=(t['linear_num_value_heads'],)
            result[a+'.dt_bias']=(t['linear_num_value_heads'],)
            result[a+'.norm.weight']=(t['linear_value_head_dim'],)
    vh=v['hidden_size']; p='model.visual'
    result[p+'.patch_embed.proj.weight']=(vh,v['in_channels'],v['temporal_patch_size'],v['patch_size'],v['patch_size'])
    result[p+'.patch_embed.proj.bias']=(vh,)
    result[p+'.pos_embed.weight']=(v['num_position_embeddings'],vh)
    for i in range(v['depth']):
        b=f'{p}.blocks.{i}'
        for norm in ['norm1','norm2']:
            result[b+'.'+norm+'.weight']=(vh,); result[b+'.'+norm+'.bias']=(vh,)
        linear(b+'.attn.qkv',vh*3,vh,True); linear(b+'.attn.proj',vh,vh,True)
        linear(b+'.mlp.linear_fc1',v['intermediate_size'],vh,True)
        linear(b+'.mlp.linear_fc2',vh,v['intermediate_size'],True)
    result[p+'.merger.norm.weight']=(vh,); result[p+'.merger.norm.bias']=(vh,)
    mh=vh*v['spatial_merge_size']**2
    linear(p+'.merger.linear_fc1',mh,mh,True); linear(p+'.merger.linear_fc2',v['out_hidden_size'],mh,True)
    return result
#!/usr/bin/env python3
"""per_layer_cosine.py — Bisect where the correlation cliff occurs.
Compares BQSM vs float reference at: embedding, layer0-attn, layer0-ffn, layer1-attn, layer1-ffn, final.
Reports cosine at each checkpoint to locate the specific step where correlation dies."""
import numpy as np
from gguf import GGUFReader

GGUF = '/home/compunerd/Downloads/gemma-2-2b-it-abliterated-Q4_K_M.gguf'
reader = GGUFReader(GGUF)

# Dequant helpers (same as before, with clip)
def dequant_q4(raw, ne):
    nb=ne//256; blk=raw[:nb*144].reshape(nb,144)
    d=blk[:,0:2].view(np.float16).flatten().astype(np.float32)
    dm=blk[:,2:4].view(np.float16).flatten().astype(np.float32)
    qs=blk[:,4:132]; lo=(qs&0x0F).astype(np.float32); hi=(qs>>4).astype(np.float32)
    nib=np.empty((nb,256),np.float32); nib[:,0::2]=lo; nib[:,1::2]=hi
    sr=blk[:,132:144].astype(np.int32)
    sc=np.zeros((nb,8),np.float32); m=np.zeros((nb,8),np.float32)
    for i in range(6): sc[:,i]=sr[:,i]&63
    sc[:,6]=(sr[:,6]&63)|((sr[:,9]&0x30)<<2); sc[:,7]=(sr[:,7]&63)|((sr[:,9]&0xC0))
    m[:,0]=sr[:,8]&63; m[:,1]=sr[:,9]&63; m[:,2]=sr[:,10]&63; m[:,3]=sr[:,11]&63
    m[:,4]=(sr[:,11]>>6)|((sr[:,10]&0x30)); m[:,5]=(sr[:,10]>>6)|((sr[:,9]&0x0C)<<2)
    m[:,6]=sr[:,9]>>6; m[:,7]=sr[:,8]>>6
    sb=np.empty((nb,256),np.float32); mb=np.empty((nb,256),np.float32)
    for s in range(8): sb[:,s*32:(s+1)*32]=sc[:,s:s+1]; mb[:,s*32:(s+1)*32]=m[:,s:s+1]
    w=(d.reshape(nb,1)*sb*(nib-mb)-dm.reshape(nb,1)).ravel()[:ne]
    return np.clip(w,-1e3,1e3)

def dequant_q6(raw, ne):
    nb=ne//256; res=np.empty(ne,np.float32)
    for st in range(0,nb,32768):
        en=min(st+32768,nb); nc=en-st
        blk=raw[st*210:en*210].reshape(nc,210)
        # Q6_K: ql[0:128], qh[128:192], sc_int8[192:208], d_fp16[208:210]
        ql=blk[:,0:128]; qh=blk[:,128:192]
        sr=blk[:,192:208].view(np.int8).astype(np.float32)
        d=blk[:,208:210].view(np.float16).flatten().astype(np.float32)
        d=np.nan_to_num(d,nan=0.0,posinf=0.0,neginf=0.0)
        lo=np.empty((nc,256),np.float32); lo[:,0::2]=(ql&0x0F).astype(np.float32); lo[:,1::2]=(ql>>4).astype(np.float32)
        hi=np.empty((nc,256),np.float32)
        for s in range(4): hi[:,s::4]=((qh>>(2*s))&0x03).astype(np.float32)
        qv=lo+hi*16.0; sb=np.empty((nc,256),np.float32)
        for s in range(16): sb[:,s*16:(s+1)*16]=sr[:,s:s+1]
        w=(d.reshape(nc,1)*sb*(qv-32.0)).ravel()
        res[st*256:en*256]=np.clip(w,-1e3,1e3)
    return np.clip(res,-1e3,1e3)

def load_tensor(name):
    for t in reader.tensors:
        if t.name==name:
            raw=np.array(t.data.flat,dtype=np.uint8); shape=tuple(int(d) for d in t.shape)
            ne=int(np.prod(shape)); dt=t.tensor_type.name
            if dt=='F32': w=np.array(t.data.flat,dtype=np.float32)[:ne]
            elif dt=='Q4_K': w=dequant_q4(raw,ne)
            elif dt=='Q6_K': w=dequant_q6(raw,ne)
            else: continue
            return w.reshape(shape).astype(np.float32)
    return None

print('Loading weights for 2-layer float reference...')
emb=load_tensor('token_embd.weight')
D,HD,NQ,NKV=2304,256,8,4
layers=[{s:load_tensor(f'blk.{L}.{s}') for s in ['attn_norm.weight','attn_q.weight','attn_k.weight','attn_v.weight','attn_output.weight','ffn_norm.weight','ffn_gate.weight','ffn_up.weight','ffn_down.weight']} for L in range(2)]
on=load_tensor('output_norm.weight')

def rms_norm(x,w): return x*w/np.sqrt(np.mean(x**2)+1e-6)
def softmax(x): e=np.exp(x-np.max(x,axis=-1,keepdims=True)); return e/np.sum(e,axis=-1,keepdims=True)

# 'the cat': IDs [2, 1175, 4401]
tids=[2,1175,4401]
checkpoints = {}
x=np.zeros((len(tids),D),np.float32)
for i,tid in enumerate(tids): 
    x[i]=emb[:,tid]*np.sqrt(D)
    checkpoints[f'embed[{i}]'] = x[i].copy()

kvc=[{'k':[],'v':[]} for _ in range(2)]
for pos in range(len(tids)):
    h=x[pos:pos+1]
    checkpoints[f'L0_in[{pos}]'] = h[0].copy()
    
    for L in range(2):
        lw=layers[L]; hn=rms_norm(h,lw['attn_norm.weight'])
        q=hn@lw['attn_q.weight']; k=hn@lw['attn_k.weight']; v=hn@lw['attn_v.weight']
        q=q.reshape(1,NQ,HD); k=k.reshape(1,NKV,HD); v=v.reshape(1,NKV,HD)
        kvc[L]['k'].append(k[0]); kvc[L]['v'].append(v[0])
        kc=np.stack(kvc[L]['k']); vc=np.stack(kvc[L]['v'])
        ao=np.zeros((1,NQ,HD),np.float32)
        for qh in range(NQ):
            kvh=qh//2; sc=q[0,qh]@kc[:,kvh].T/np.sqrt(HD); aw=softmax(sc); ao[0,qh]=aw@vc[:,kvh]
        af=ao.reshape(1,NQ*HD); attn_out=h+af@lw['attn_output.weight']
        checkpoints[f'L{L}_attn[{pos}]'] = attn_out[0].copy()
        hn2=rms_norm(attn_out,lw['ffn_norm.weight'])
        gate=hn2@lw['ffn_gate.weight']; up=hn2@lw['ffn_up.weight']
        gelu=lambda x:0.5*x*(1.0+np.tanh(np.sqrt(2.0/np.pi)*(x+0.044715*x**3)))
        fo=gelu(gate)*up; h=attn_out+fo@lw['ffn_down.weight']
        checkpoints[f'L{L}_ffn[{pos}]'] = h[0].copy()
    
    if pos==len(tids)-1: 
        final=rms_norm(h,on)
        checkpoints['final'] = final[0].copy()

# Print checkpoint stats
print('\nFloat reference checkpoints:')
for name, vec in checkpoints.items():
    print(f'  {name:20s}: mean={vec.mean():+.4f}  std={vec.std():.4f}  range=[{vec.min():+.3f},{vec.max():+.3f}]')

# Save for later comparison with BQSM
np.savez('/tmp/float_checkpoints.npz', **checkpoints)
print('\nSaved /tmp/float_checkpoints.npz')
print('\nNow add same checkpoints to BQSM forward_pass and capture them.')

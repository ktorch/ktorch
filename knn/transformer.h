#pragma once
#include "util.h"
#include "norm.h"

namespace knn {

// ---------------------------------------------------------------------------
// encode/decode layers - set/get options for transformer encode/decode layers
// ---------------------------------------------------------------------------
// as of v1.10.0, custom function added on end of variant of `relu`gelu, complicating retrieval
// activation_t variant<enumtype::kReLU, enumtype::kGELU, std::function<Tensor(const Tensor&)> >

torch::nn::activation_t codefn(Cast,S);
S codefn(Cast,const torch::nn::activation_t&);

template<typename O>O codelayer(K x,J i,Cast c) {
 TORCH_CHECK(x->t>=0 || x->t==99, msym(c),": unrecognized  or insufficient arg(s), ",kname(x),", ",kstring(x));
 Pairs p; J n=xargc(x,i,p); O o(0,0);
 for(J j=0;j<n;++j) {
  switch(j) {
   case 0: o.d_model(int64(x,i+j,c,Setting::in)); break;
   case 1: o.nhead(int64(x,i+j,c,Setting::heads)); break;
   case 2: o.dim_feedforward(int64(x,i+j,c,Setting::dim)); break;
   case 3: o.dropout(mdouble(x,i+j,c,Setting::dropout)); break;
   case 4: o.activation(codefn(c,code(x,i+j,c,Setting::fn))); break;
   default: mpos(x,c,i+j); break;
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:      o.d_model(int64(p,c)); break;
   case Setting::heads:   o.nhead(int64(p,c)); break;
   case Setting::dim:     o.dim_feedforward(int64(p,c)); break;
   case Setting::dropout: o.dropout(mdouble(p,c)); break;
   case Setting::fn:      o.activation(codefn(c,code(p,c))); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.d_model()>0, msym(c), ": positive number of input features required");
 TORCH_CHECK(  o.nhead()>0, msym(c), ": positive number of heads required");
 return o;
}

template<typename O>K codelayer(bool a,Cast c,const O& o) {
 K x=KDICT; O d(o.d_model(),o.nhead());
 msetting(x, Setting::in,    kj(o.d_model()));
 msetting(x, Setting::heads, kj(o.nhead()));
 if(a || (o.dim_feedforward()    != d.dim_feedforward()))    msetting(x, Setting::dim,     kj(o.dim_feedforward()));
 if(a || (o.dropout()            != d.dropout()))            msetting(x, Setting::dropout, kf(o.dropout()));
 if(a || (o.activation().index() != d.activation().index())) msetting(x, Setting::fn,      ks(codefn(c,o.activation())));
 return x;
}

// ------------------------------------------------------------------------------------
// create transformer encoder/decoder layers:
// codeoff - offset for parsing submodule arg(s), e.g. (`encodelayer;512;8) -> offset=1
// codelayer - template to process a created layer module or the args to define one
// codenorm - process a created layernorm module or the arg(s) needed to define one
// coder - template for encoder/decoder layers w'submodule, layer count, optional norm 
// encoder,decoder - invoke template 'coder' with encoder/decoder layer & option types
// -----------------------------------------------------------------------------------
J codeoff(K,Cast);
 
template<typename L,typename O> L codelayer(K x,Cast c,std::vector<K>& v) {
 Kmodule *m=xmodule(x); bool e=c == Cast::encoder;
 if(m) {
  auto *l=m->m->as<L>();
  TORCH_CHECK(l, msym(c),": expecting ",(e ? "encoding" : "decoding")," layer, given ",mlabel(m)," module");
  v.push_back(x);
  return L(*l);
 } else {
  return L(codelayer<O>(x,codeoff(x,c),e ? Cast::encoderlayer : Cast::decoderlayer));
 }
}

torch::nn::LayerNorm codenorm(K,J,Cast,std::vector<K>&);

template<typename R,typename L,typename O> R coder(K x,J i,Cast c) {
 TORCH_CHECK(x->t==0 || x->t==99, msym(c),": unrecognized  or insufficient arg(s), ",kname(x),", ",kstring(x));
 Pairs p; J l=-1,n=xargc(x,i,p); L m1=nullptr; torch::nn::LayerNorm m2=nullptr; std::vector<K> v;
 for(J j=0;j<n;++j) {
  K y=kK(x)[i+j];
  switch(j) {
   case 0: m1=codelayer<L,O>(y,c,v); break;
   case 1: l=int64(x,i+j,c,Setting::layers); break;
   case 2: if(!xempty(y)) m2=codenorm(x,0,c,v); break;
   default: mpos(x,c,i+j); break;
  }
 }
 while(xpair(p)) {
  switch(mset(p.k,c)) {
   case Setting::decoderlayer:
    TORCH_CHECK(c==Cast::decoder, msym(c),": cannot create a decoder layer");
    TORCH_CHECK(p.t>=0, msym(c),": unrecognized arg(s) for decoder layer (",kname(p.t),")");
    m1=codelayer<L,O>(p.v,c,v);
    break;
   case Setting::encoderlayer:
    TORCH_CHECK(c==Cast::encoder, msym(c),": cannot create an encoder layer");
    TORCH_CHECK(p.t>=0, msym(c),": unrecognized arg(s) for encoder layer (",kname(p.t),")");
    m1=codelayer<L,O>(p.v,c,v);
    break;
   case Setting::layers:
    l=int64(p,c);
    break;
   case Setting::layernorm:
    if(!pempty(p)) {
     TORCH_CHECK(p.t==-KJ || p.t>=0, msym(c),": unrecognized arg(s) for layer normalization (",kname(p.t),")");
     m2=codenorm(p.t==-KJ ? nullptr : p.v, p.j, c, v);
    }
    break;
   default: mpair(c,p); break;
  }
 }
 TORCH_CHECK(l>=0, msym(c), ": non-negative number of layers must be defined");
 auto r=m2.is_empty() ? R(m1,l) : R(m1,l).norm(AnyModule(m2));
 if(v.size()) kfree(v);
 return r;
}

torch::nn::TransformerDecoderOptions decoder(K,J,Cast);
torch::nn::TransformerEncoderOptions encoder(K,J,Cast);

// -----------------------------------------------------------------------------
// codenorm - retrieve dictionary of options in layer norm module if not empty
// coder - templated retrieval of common options of encoder/decoder layer
// decoder,encoder - retrieve options of encoder/decoder layer,layer count,norm
// -----------------------------------------------------------------------------
K codenorm(bool,const AnyModule&);

template<typename O> void coder(bool a,K x,const O& o) {
 msetting(x, Setting::layers,    kj(o.num_layers()));
 msetting(x, Setting::layernorm, codenorm(a,o.norm()));
}

K decoder(bool,Cast,const torch::nn::TransformerDecoderOptions&);
K encoder(bool,Cast,const torch::nn::TransformerEncoderOptions&);

// -------------------------------------------------------------------------------------
// customcoder - create or retrieve options from custom encoder/decoder for transformer
// transformer - create transformer or retrieve options from existing transformer module
// -------------------------------------------------------------------------------------
AnyModule customcoder(K,Setting,std::vector<K>&);
torch::nn::TransformerOptions transformer(K,J,Cast);
K customcoder(bool,const AnyModule&);
K transformer(bool,Cast,const torch::nn::TransformerOptions&);

} // namespace knn

#include "../ktorch.h"
#include "transformer.h"

namespace knn {

// ---------------------------------------------------------------------------
// encode/decode layers - set/get options for transformer encode/decode layers
// ---------------------------------------------------------------------------
// as of v1.10.0, custom function added on end of variant of `relu`gelu, complicating retrieval
// activation_t variant<enumtype::kReLU, enumtype::kGELU, std::function<Tensor(const Tensor&)> >

torch::nn::activation_t codefn(Cast c,S s) {
 switch(emap(s)) {
  case Enum::relu: return torch::kReLU;
  case Enum::gelu: return torch::kGELU;
  default: TORCH_ERROR("unrecognized ", msym(c), " activation fn: `",s); break;
 }
}

S codefn(Cast c,const torch::nn::activation_t& f) {
 if(c10::get_if<torch::enumtype::kReLU>(&f)) {
  return std::get<0>(env().enums[(size_t)Enum::relu]);
 } else if(c10::get_if<torch::enumtype::kGELU>(&f)) {
  return std::get<0>(env().enums[(size_t)Enum::gelu]);
 } else {
  TORCH_ERROR(msym(c),": unable to extract custom activation function");
 }
}

// ------------------------------------------------------------------------------------
// create transformer encoder/decoder layers:
// codeoff - offset for parsing submodule arg(s), e.g. (`encodelayer;512;8) -> offset=1
// codenorm - process a created layernorm module or the arg(s) needed to define one
// encoder,decoder - invoke template 'coder' with encoder/decoder layer & option types
// -----------------------------------------------------------------------------------
J codeoff(K x,Cast c) {
 J i=0; S s=nullptr;
 switch(x->t) {
  case 0:   if(xsym(x,0,s)) i=xsym(x,1) ? 2 : 1; break;
  case -KS: s=x->s, i=1; break;
  case KS:  if(x->n) s=kS(x)[0], i=(2<x->n) ? 2 : x->n; break;
  case 99:  i=-1; break;
 }
 if(s) { // module sym not required; if given, check if one of `encoderlayer`decoderlayer`layernorm
  auto m=msym(s);
  TORCH_CHECK(m==Cast::layernorm || m==(c==Cast::encoder ? Cast::encoderlayer : Cast::decoderlayer),
              msym(c), ": unexpected layer '", s, "'");
 }
 return i;
}
 
torch::nn::LayerNorm codenorm(K x,J n,Cast c,std::vector<K>& v) {
 if(x) {
  auto *m=xmodule(x);
  if(m) {
   auto *l=m->m->as<torch::nn::LayerNorm>();
   TORCH_CHECK(l, msym(c),": expecting normalization layer, given ",mlabel(m)," module");
   v.push_back(x);
   return torch::nn::LayerNorm(*l);
  } else {
   return torch::nn::LayerNorm(x->t==-KJ ? torch::nn::LayerNormOptions({x->j}) : knn::layernorm(x,codeoff(x,c),Cast::layernorm));
  }
 } else {
  return torch::nn::LayerNorm(torch::nn::LayerNormOptions({n}));
 }
}

torch::nn::TransformerDecoderOptions decoder(K x,J i,Cast c) {
 return coder<torch::nn::TransformerDecoderOptions,
              torch::nn::TransformerDecoderLayer,  
              torch::nn::TransformerDecoderLayerOptions>(x,i,c);
}  

torch::nn::TransformerEncoderOptions encoder(K x,J i,Cast c) {
 return coder<torch::nn::TransformerEncoderOptions,
              torch::nn::TransformerEncoderLayer,  
              torch::nn::TransformerEncoderLayerOptions>(x,i,c);
}  

// -----------------------------------------------------------------------------
// codenorm - retrieve dictionary of options in layer norm module if not empty
// decoder,encoder - retrieve options of encoder/decoder layer,layer count,norm
// -----------------------------------------------------------------------------
K codenorm(bool a,const AnyModule& m) {
 return m.is_empty() ? KDICT : knn::layernorm(a,m.get<torch::nn::LayerNorm>()->options);
}

K decoder(bool a,Cast c,const torch::nn::TransformerDecoderOptions& o) {
 K x=KDICT; msetting(x, Setting::decoderlayer, codelayer(a,c,o.decoder_layer()->options)); coder(a,x,o); return x;
}

K encoder(bool a,Cast c,const torch::nn::TransformerEncoderOptions& o) {
 K x=KDICT; msetting(x, Setting::encoderlayer, codelayer(a,c,o.encoder_layer()->options)); coder(a,x,o); return x;
}

// -------------------------------------------------------------------------------------
// anymodule - call main AnyModule creation function after mapping symbol to module type
// customcoder - create or retrieve options from custom encoder/decoder for transformer
// transformer - create transformer or retrieve options from existing transformer module
// -------------------------------------------------------------------------------------
static AnyModule anymodule(K x,J i,S s) {
 Cast c=msym(s);
 return anymodule(c, mcreate(x,i,c));
}

AnyModule customcoder(K x,Setting t,std::vector<K>& v) {
 K y; J i; S s,nm; AnyModule a; Kmodule *m=xmodule(x);
 if(m) {
  a=anymodule(m->c,m->m);
  v.push_back(x);
 } else {
  if(xdict(x)) {
   i=-1; s=statemodule(x); nm=statename(x), y=stateoptions(x);
  } else {
   y=x; msyms(y,s,nm); i=argstart(y,nm);
  }
  a=anymodule(y,i,s);
 }
 return a;
}

torch::nn::TransformerOptions transformer(K x,J i,Cast c) {
 Pairs p; Setting s; J n=xargc(x,i,p); torch::nn::TransformerOptions o; std::vector<K> v;
 for(J j=0;j<n;++j) {
  switch(j) {
   case 0: o.d_model(int64(x,i+j,c,Setting::in)); break;
   case 1: o.nhead(int64(x,i+j,c,Setting::heads)); break;
   case 2: o.num_encoder_layers(int64(x,i+j,c,Setting::elayers)); break;
   case 3: o.num_decoder_layers(int64(x,i+j,c,Setting::dlayers)); break;
   case 4: o.dim_feedforward(int64(x,i+j,c,Setting::dim)); break;
   case 5: o.dropout(mdouble(x,i+j,c,Setting::dropout)); break;
   case 6: o.activation(codefn(c,code(x,i+j,c,Setting::fn))); break;
   case 7: if(!xempty(x,i+j)) o.custom_encoder(customcoder(kK(x)[i+j],Setting::encoder,v)); break;
   case 8: if(!xempty(x,i+j)) o.custom_decoder(customcoder(kK(x)[i+j],Setting::decoder,v)); break;
   default: mpos(x,c,i+j); break;
  }
 }
 while(xpair(p))
  switch((s=mset(p.k,c))) {
   case Setting::in:      o.d_model(int64(p,c)); break;
   case Setting::heads:   o.nhead(int64(p,c)); break;
   case Setting::elayers: o.num_encoder_layers(int64(p,c)); break;
   case Setting::dlayers: o.num_decoder_layers(int64(p,c)); break;
   case Setting::dim:     o.dim_feedforward(int64(p,c)); break;
   case Setting::dropout: o.dropout(mdouble(p,c)); break;
   case Setting::fn:      o.activation(codefn(c,code(p,c))); break;
   case Setting::encoder: 
   case Setting::decoder:
    if(!pempty(p)) {
     TORCH_CHECK(p.t==-KS || p.t>=0, msym(c), ": unrecognized arg(s) for custom ",p.k);
     auto a=p.t==-KS ? anymodule(nullptr,0,p.s) : customcoder(p.v,s,v);
     s==Setting::encoder ?  o.custom_encoder(a) : o.custom_decoder(a);
    }
    break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.d_model()>0, msym(c), ": positive number of input features required");
 TORCH_CHECK(  o.nhead()>0, msym(c), ": positive number of heads required");
 if(v.size()) kfree(v);  // free any allocated submodules
 return o;
}

K customcoder(bool a,const AnyModule& y) {
 K k=ktn(KS,2),v=ktn(0,2); const Module& m=*y.ptr(); Cast c=mcast(m);
 kS(k)[0]=statekey(State::module);  kK(v)[0]=ks(msym(c));
 kS(k)[1]=statekey(State::options); kK(v)[1]=moduleoptions(a,false,c,m);
 return xD(k,v);
}

K transformer(bool a,Cast c,const torch::nn::TransformerOptions& o) {
 K x=KDICT; torch::nn::TransformerOptions d;
 if(a || (o.d_model()            != d.d_model()))            msetting(x, Setting::in,      kj(o.d_model()));
 if(a || (o.nhead()              != d.nhead()))              msetting(x, Setting::heads,   kj(o.nhead()));
 if(a || (o.num_encoder_layers() != d.num_encoder_layers())) msetting(x, Setting::elayers, kj(o.num_encoder_layers()));
 if(a || (o.num_decoder_layers() != d.num_decoder_layers())) msetting(x, Setting::dlayers, kj(o.num_decoder_layers()));
 if(a || (o.dim_feedforward()    != d.dim_feedforward()))    msetting(x, Setting::dim,     kj(o.dim_feedforward()));
 if(a || (o.dropout()            != d.dropout()))            msetting(x, Setting::dropout, kf(o.dropout()));
 if(a || (o.activation().index() != d.activation().index())) msetting(x, Setting::fn,      ks(codefn(c,o.activation())));
 if(o.custom_encoder().is_empty()) {
  if(a) msetting(x, Setting::encoder, KDICT);
 } else { 
  msetting(x, Setting::encoder, customcoder(a,o.custom_encoder()));
 }
 if(o.custom_decoder().is_empty()) {
  if(a) msetting(x, Setting::decoder, KDICT);
 } else { 
  msetting(x, Setting::decoder, customcoder(a,o.custom_decoder()));
 }
 return x;
}

} // namespace knn

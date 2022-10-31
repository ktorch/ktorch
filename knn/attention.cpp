#include "../ktorch.h"
#include "attention.h"
#include "norm.h"

namespace knn {

// ------------------------------------------------------------------------
// SelfAttention - create base MultiheadAttention w'options & optional norm
// ------------------------------------------------------------------------
SelfAttentionImpl::SelfAttentionImpl(const SelfAttentionOptions& o) : options(o) {
 reset();;
}

void SelfAttentionImpl::reset() {
 if(options.norm())
  norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({options.dim()})));
 in   = register_module("in",   torch::nn::Linear(torch::nn::LinearOptions(options.dim(),options.dim()*3).bias(!options.norm())));
 drop = register_module("drop", torch::nn::Dropout(options.dropout()));
 out  = register_module("out",  torch::nn::Linear(options.dim(), options.dim()));
}

void SelfAttentionImpl::pretty_print(std::ostream& s) const {
 s << "knn::SelfAttention(dim=" 
   << options.dim() << ", heads="
   << options.heads() << ", dropout="
   << options.dropout()   << ", norm="
   << std::boolalpha << options.norm() << ")";
}

Tensor SelfAttentionImpl::forward(const Tensor& x,const Tensor& m,const Tensor& p) {
 using namespace torch::indexing;
 const auto& s=x.sizes(); auto h=options.heads(); auto hd=s[2]/h;
 const auto c=in(norm.is_empty() ? x : norm(x)).reshape({s[0],s[1],3,h,hd}).permute({2,0,3,1,4});
 auto q=c[0]; auto k=c[1]; auto v=c[2];
 auto a=torch::matmul(q, k.transpose(-2, -1)) * (1 / std::sqrt(hd));
 if(m.defined())
  a += a.size(-1)==m.size(-1) ? m : m.index({Slice(None,a.size(-1)),Slice(None,a.size(-1))});
 TORCH_CHECK(!p.defined(), "self attention: padding mask not implemented");
 a=torch::matmul(drop(torch::softmax(a,-1)), v);
 return out(a.transpose(1,2).contiguous().view(s));
}

// -----------------------------------------------------
// selfattn - set/get options for self attention
// -----------------------------------------------------
SelfAttentionOptions selfattn(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); SelfAttentionOptions o(0,0);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.dim(int64(x,i+j,c,Setting::dim)); break;
    case 1: o.heads(int64(x,i+j,c,Setting::heads)); break;
    case 2: o.dropout(mdouble(x,i+j,c,Setting::dropout)); break;
    case 3: o.norm(mbool(x,i+j,c,Setting::norm)); break;
    default: mpos(x,c,i+j); break;
   }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim:       o.dim(int64(p,c)); break;
   case Setting::heads:     o.heads(int64(p,c)); break;
   case Setting::dropout:   o.dropout(mdouble(p,c)); break;
   case Setting::norm:      o.norm(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.dim()>0, msym(c), ": positive embedding dimension required");
 TORCH_CHECK(o.heads()>0, msym(c), ": positive number of heads required");
 TORCH_CHECK(!(o.dim() % o.heads()) && o.heads()<o.dim(), 
             msym(c), ": model dimension of ",o.dim()," not divisible by ",o.heads()," heads");
 return o;
}

K selfattn(bool a,const SelfAttentionOptions& o) {
 K x=KDICT; SelfAttentionOptions d(o.dim(),o.heads());
 msetting(x, Setting::dim,   kj(o.dim()));
 msetting(x, Setting::heads, kj(o.heads()));
 if(a || (o.dropout() != d.dropout())) msetting(x, Setting::dropout, kf(o.dropout()));
 if(a || (o.norm()    != d.norm()))    msetting(x, Setting::norm,    kb(o.norm()));
 return resolvedict(x);
}

// -----------------------------------------------------
// attention - set/get options for multi-head attention
// -----------------------------------------------------
torch::nn::MultiheadAttentionOptions attention(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); torch::nn::MultiheadAttentionOptions o(0,0);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.embed_dim(int64(x,i+j,c,Setting::dim)); break;
    case 1: o.num_heads(int64(x,i+j,c,Setting::heads)); break;
    case 2: o.dropout(mdouble(x,i+j,c,Setting::dropout)); break;
    case 3: o.bias(mbool(x,i+j,c,Setting::bias)); break;
    case 4: o.add_bias_kv(mbool(x,i+j,c,Setting::addbias)); break;
    case 5: o.add_zero_attn(mbool(x,i+j,c,Setting::addzero)); break;
    case 6: o.kdim(int64(x,i+j,c,Setting::kdim)); break;
    case 7: o.vdim(int64(x,i+j,c,Setting::vdim)); break;
    default: mpos(x,c,i+j); break;
   }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim:     o.embed_dim(int64(p,c)); break;
   case Setting::heads:   o.num_heads(int64(p,c)); break;
   case Setting::dropout: o.dropout(mdouble(p,c)); break;
   case Setting::bias:    o.bias(mbool(p,c)); break;
   case Setting::addbias: o.add_bias_kv(mbool(p,c)); break;
   case Setting::addzero: o.add_zero_attn(mbool(p,c)); break;
   case Setting::kdim:    o.kdim(int64(p,c)); break;
   case Setting::vdim:    o.vdim(int64(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.embed_dim()>0, msym(c), ": positive embedding dimension required");
 TORCH_CHECK(o.num_heads()>0, msym(c), ": positive number of heads required");
 if(o.kdim()<=0) o.kdim(o.embed_dim());
 if(o.vdim()<=0) o.vdim(o.embed_dim());
 return o;
}

K attention(bool a,const torch::nn::MultiheadAttentionOptions& o) {
 K x=KDICT; torch::nn::MultiheadAttentionOptions d(o.embed_dim(),o.num_heads());
 msetting(x, Setting::dim,   kj(o.embed_dim()));
 msetting(x, Setting::heads, kj(o.num_heads()));
 if(a || (o.dropout()       != d.dropout()))       msetting(x, Setting::dropout, kf(o.dropout()));
 if(a || (o.bias()          != d.bias()))          msetting(x, Setting::bias,    kb(o.bias()));
 if(a || (o.add_bias_kv()   != d.add_bias_kv()))   msetting(x, Setting::addbias, kb(o.add_bias_kv()));
 if(a || (o.add_zero_attn() != d.add_zero_attn())) msetting(x, Setting::addzero, kb(o.add_zero_attn()));
 if(a || (o.kdim()          != d.kdim()))          msetting(x, Setting::kdim,    kj(o.kdim()));
 if(a || (o.vdim()          != d.vdim()))          msetting(x, Setting::vdim,    kj(o.vdim()));
 return resolvedict(x);
}

} // namespace knn

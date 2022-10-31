#include "../ktorch.h"
#include "reshape.h"

namespace knn {

// -------------------------------------------------------------
// expand, reshape & permute - modules with size options
// -------------------------------------------------------------
ExpandImpl::ExpandImpl(std::vector<int64_t> s) : ExpandImpl(SizeOptions(s)) {}
ExpandImpl::ExpandImpl(const SizeOptions& o) : options(o) {reset();}
void ExpandImpl::reset() {}
void ExpandImpl::pretty_print(std::ostream& s) const {
 s << "knn::Expand(size=" << options.size() << ")";
}
Tensor ExpandImpl::forward(const Tensor& x) {
 return x.expand(options.size());
}

ReshapeImpl::ReshapeImpl(std::vector<int64_t> s) : ReshapeImpl(SizeOptions(s)) {}
ReshapeImpl::ReshapeImpl(const SizeOptions& o) : options(o) {reset();}
void ReshapeImpl::reset() {}
void ReshapeImpl::pretty_print(std::ostream& s) const {
 s << "knn::Reshape(size=" << options.size() << ")";
}
Tensor ReshapeImpl::forward(const Tensor& x) {
 return x.reshape(options.size());
}

PermuteImpl::PermuteImpl(std::vector<int64_t> s) : PermuteImpl(SizeOptions(s)) {}
PermuteImpl::PermuteImpl(const SizeOptions& o) : options(o) {reset();}
void PermuteImpl::reset() {}
void PermuteImpl::pretty_print(std::ostream& s) const {
 s << "knn::Permute(size=" << options.size() << ")";
}
Tensor PermuteImpl::forward(const Tensor& x) {
 return x.permute(options.size());
}

// -------------------------------------------------------------
// transpose - transpose given 2 dimensions
// -------------------------------------------------------------
TransposeImpl::TransposeImpl(const TransposeOptions& o) : options(o) {reset();}
void TransposeImpl::reset() {}
void TransposeImpl::pretty_print(std::ostream& s) const {
 s << "knn::Transpose(dim0=" << options.dim0() << ", dim1=" << options.dim1() << ")";
}
Tensor TransposeImpl::forward(const Tensor& x) {
 return x.transpose(options.dim0(),options.dim1());
}

// ----------------------------------------------
// getsize - set/get size(s) for expand & reshape
// ----------------------------------------------
SizeOptions getsize(K x,J i,Cast c) {
 SizeOptions o({}); Pairs p; J n=xargc(x,i,p);
 TORCH_CHECK(n<2, msym(c),": 1 positional argument expected, ",n," given");
 if(n==1) o.size(mlongs(x,i,c,Setting::size));
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size: o.size(mlongs(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

K getsize(bool a,const SizeOptions& o) {
 K x=KDICT;
 msetting(x, Setting::size, klist(o.size().size(),o.size().data()));
 return resolvedict(x);
}

// ----------------------------------------
// flatten - get/set start & end dimensions
// ----------------------------------------
torch::nn::FlattenOptions flatten(K x,J i,Cast c,bool f) {
 torch::nn::FlattenOptions o; if(f) o.start_dim(0); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.start_dim(int64(x,i+j,c,Setting::start));  break;
    case 1: o.end_dim(int64(x,i+j,c,Setting::end));  break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::start: o.start_dim(int64(p,c)); break;
   case Setting::end:   o.end_dim(int64(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

K flatten(bool a,const torch::nn::FlattenOptions& o) {
 K x=KDICT; torch::nn::FlattenOptions d;
 if(a || d.start_dim() != o.start_dim()) msetting(x, Setting::start, kj(o.start_dim()));
 if(a || d.end_dim()   != o.end_dim())   msetting(x, Setting::end,   kj(o.end_dim()));
 return resolvedict(x);
}

// ----------------------------------------
// transpose - get/set dim0 & dim1
// ----------------------------------------
TransposeOptions transpose(K x,J i,Cast c) {
 TransposeOptions o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.dim0(int64(x,i+j,c,Setting::dim0));  break;
    case 1: o.dim1(int64(x,i+j,c,Setting::dim1));  break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim0: o.dim0(int64(p,c)); break;
   case Setting::dim1: o.dim1(int64(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

K transpose(bool a,const TransposeOptions& o) {
 K x=KDICT;
 msetting(x, Setting::dim0, kj(o.dim0()));
 msetting(x, Setting::dim1,   kj(o.dim1()));
 return resolvedict(x);
}

} // namespace knn

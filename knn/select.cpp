#include "../ktorch.h"
#include "select.h"

namespace knn {

// -------------------------------------------------------
// select - module for tensor.select(dim,ind)
// -------------------------------------------------------
SelectImpl::SelectImpl(const SelectOptions& o) : options(o) {reset();}

void SelectImpl::reset() {}

void SelectImpl::pretty_print(std::ostream& s) const {
 s << "knn::Select(dim=" << options.dim() << ",ind=" << options.ind() << ")";
}

Tensor SelectImpl::forward(const Tensor& x) {
 return x.select(options.dim(),options.ind());
}

// -------------------------------------------------------
// indexselect - module for tensor.index_select(,dim,ind)
// -------------------------------------------------------
IndexSelectImpl::IndexSelectImpl(const IndexSelectOptions& o) : options(o) {reset();}

void IndexSelectImpl::reset() {
 TORCH_CHECK(options.ind().dtype() == torch::kLong, "select: long(s) expected for indices, ",options.ind().dtype(),"(s) supplied");
 TORCH_CHECK(options.ind().dim()<2, "select: single index or list expected, ",options.ind().dim(),"-d tensor supplied");
 ind=register_buffer("ind", options.ind());
}

void IndexSelectImpl::pretty_print(std::ostream& s) const {
 s << "knn::IndexSelect(dim=" << options.dim() << ",ind=";
 print_tensor(s,3,options.ind());
 s << ")";
}

Tensor IndexSelectImpl::forward(const Tensor& x) {
 return x.index_select(options.dim(),ind);
}

// ----------------------------------------------------------------
// select - get/set dim & scalar index for Select module
// ----------------------------------------------------------------
SelectOptions select(K x,J i,Cast c) {
 SelectOptions o(nj,nj); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.dim(int64(x,i+j,c,Setting::dim));  break;
    case 1: o.ind(int64(x,i+j,c,Setting::ind));  break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim: o.dim(int64(p,c)); break;
   case Setting::ind: o.ind(int64(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.dim() != nj, msym(c),": no dimension defined");
 TORCH_CHECK(o.ind() != nj, msym(c),": no index defined");
 return o;
}

K select(bool a,const SelectOptions& o) {
 K x=KDICT;
 msetting(x, Setting::dim, kj(o.dim()));
 msetting(x, Setting::ind, kj(o.ind()));
 return resolvedict(x);
}

// ----------------------------------------------------------------
// indexselect - get/set dim & tensor index for IndexSelect module
// ----------------------------------------------------------------
IndexSelectOptions indexselect(K x,J i,Cast c) {
 IndexSelectOptions o(nj,Tensor()); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.dim(int64(x,i+j,c,Setting::dim));  break;
    case 1: o.ind(ltensor(x,i+j,c,Setting::ind));  break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim: o.dim(int64(p,c)); break;
   case Setting::ind: o.ind(ltensor(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.dim() != nj,     msym(c),": dimension cannot be null");
 TORCH_CHECK(o.ind().defined(), msym(c),": no index defined");
 TORCH_CHECK(o.ind().dim()<2,   msym(c),": expecting scalar index or 1-dim list, given ",o.ind().dim(),"-d tensor");
 return o;
}

K indexselect(bool a,const IndexSelectOptions& o) {
 K x=KDICT;
 msetting(x, Setting::dim,   kj(o.dim()));
 msetting(x, Setting::ind, kget(o.ind()));
 return resolvedict(x);
}

} // namespace knn

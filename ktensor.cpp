#include "ktorch.h"
#include <torch/csrc/autograd/function.h>

namespace nn=torch::nn;

// ---------------------------------------------------------------------------
// kten - given tensor ref, return ptr to struct w'attrs, void ptr to tensor
// kvec - given reference to vector of tensors, return ptr to struct w'attrs
// kdict - given tensor dictionary reference, return ptr to containing struct
// ---------------------------------------------------------------------------
K kten(const Tensor& t) {return kptr(new Kten(t));}
K kvec(const TensorVector& v) {return kptr(new Kvec(v));}
K kdict(const TensorDict &d,Cast c) {return kptr(new Kdict(d,c));}

// -------------------------------------------------------------------------
// razeflag - check if general list made up entirely of scalars
// razelist - if general list is all scalars, raze to simple list
// -------------------------------------------------------------------------
static bool razeflag(K x) {
 if(!x->t && x->n>0) {
  auto t=kK(x)[0]->t;
  if(t>=0) 
   return false;
  for(J i=1; i<x->n; ++i)
   if(kK(x)[i]->t != t)
    return false;;
  return true;
 } else {
  return false;
 }
}

static K razelist(K x) {
 if(razeflag(x)) {
  J i; K y=ktn(-kK(x)[0]->t, x->n);
  switch(y->t) {
   case KE:     for(i=0; i<y->n; ++i) kE(y)[i]=kK(x)[i]->e; break;
   case KF:     for(i=0; i<y->n; ++i) kF(y)[i]=kK(x)[i]->f; break;
   case KJ:     for(i=0; i<y->n; ++i) kJ(y)[i]=kK(x)[i]->j; break;
   case KI:     for(i=0; i<y->n; ++i) kI(y)[i]=kK(x)[i]->i; break;
   case KSHORT: for(i=0; i<y->n; ++i) kH(y)[i]=kK(x)[i]->h; break;
   case KB:
   case KC:
   case KG:     for(i=0; i<y->n; ++i) kG(y)[i]=kK(x)[i]->g; break;
   default: TORCH_ERROR("unable to raze general list -> ",kname(y));
  }
  return r0(x), y;
 } else {
  return x;
 }
}

// -----------------------------------------------------------------------------------------
// cpermute - permute real representation of complex tensor from (real,'imag) -> (real;imag)
// sparsereal - convert sparse complex tensor to sparse real representation (real,'imag)
// toreal - convert complex tensor to real representation, 1 complex number to real & imag
// -----------------------------------------------------------------------------------------
static Tensor cpermute(const Tensor& x) {
 std::vector<int64_t> d;
 for(int64_t i=0; i<x.dim(); ++i) d.push_back(i-1);
 return x.permute(d);
}

static Tensor sparsereal(const Tensor& t) {
 // asof version 1.8.1, cannot go from sparse complex -> dense complex
 // so attempt sparse complex -> sparse real representation -> dense real representation
 auto n=t.sizes().vec(); n.push_back(2);
 return torch::sparse_coo_tensor(t._indices(),torch::view_as_real(t._values()),n);
}

static Tensor toreal(const Tensor& t, c10::optional<bool> b=c10::nullopt);
static Tensor toreal(const Tensor& t, c10::optional<bool> b) {
 bool c=b ? *b : env().complexfirst;
 return c ? cpermute(t.is_sparse() ? sparsereal(t).to_dense() : torch::view_as_real(t.resolve_conj()))
          :          t.is_sparse() ? sparsereal(t).to_dense() : torch::view_as_real(t.resolve_conj());
}

// -------------------------------------------------------------------------
// kgetscalar - return k scalar given a scalar tensor
// kgets - process tensor at depth, creating k array
// kget - take tensor reference, return k scalar/array
//      - take reference to vector of longs/doubles, return k list
//      - take reference to vector/deque of tensors, return k lists
//      - take reference to dictionary of tensors, return k dictionary
// -------------------------------------------------------------------------
static K kgetscalar(const Tensor &t){
 auto s=t.item();
 switch(t.scalar_type()) {
  case torch::kFloat:  return ke(s.toFloat());
  case torch::kDouble: return kf(s.toDouble());
  case torch::kHalf:   return ke(s.toFloat());
  case torch::kShort:  return kh(s.toShort());
  case torch::kInt:    return ki(s.toInt());
  case torch::kLong:   return kj(s.toLong());
  case torch::kBool:   return kb(s.toBool());
  case torch::kByte:   return kg(s.toByte());
  case torch::kChar:   return kc(s.toChar());
  default: TORCH_ERROR("unrecognized scalar tensor type: ", t.dtype(), ", cannot return k scalar"); return (K)0;
 }
}

static K kgets(I i,I j,Ktype k,J b,const int64_t *s,S &p) {
//i:depth, j:max depth, k:k type, b:bytes to copy, s:sizes, p:data ptr
 K x=ktn((i<j) ? 0 : k,s[i]);                     //create k list
 if(x->t) {                                       //if base type
  if(x->n) {                                      //   and non-zero length
   memcpy(kG(x),p,b);                             //copy k <- tensor
   p+=b;                                          // and incr data ptr
  }
 } else {                                         // else
   for(J y=0;y<x->n;++y)                          // call down a level
    kK(x)[y]=kgets(i+1,j,k,b,s,p);                // until base data type
 }
 return x;
}

K kget(const Tensor &t) {
 if(!t.defined())
  return knull();
 else if (t.is_complex())
  return kget(toreal(t));
 else if (!t.dim())      // if 0-dimensional
  return kgetscalar(t);  // return scalar
 Tensor c;
 if(t.dtype()==torch::kHalf)
  c=t.toType(torch::kFloat).contiguous().toBackend(torch::Backend::CPU);
 else if (t.layout()==torch::kSparse)
  c=t.to_dense().toBackend(torch::Backend::CPU);
 else
  c=t.contiguous().toBackend(torch::Backend::CPU);
 I j=c.dim()-1; const int64_t *s=c.sizes().data();  // dimension & sizes at each dim
 J b=s[j]*c.element_size();                   // bytes to copy at lowest depth
 S p=(S)c.data_ptr();                         // contiguous data pointer
 return kgets(0,j,maptype(t.dtype()),b,s,p);
}

K kget(const LongVector& v)   {return klist(v.size(),v.data());}
K kget(const DoubleVector& v) {return klist(v.size(),v.data());}

K kget(const TensorDeque& v) {
 bool b=true;
 for(const auto& t:v)
  if(t.dim()) {
   b=false;
   break;
  }
 if(b) std::cerr << "deque is all scalars..\n";
 K x=ktn(0,v.size());
 for(size_t i=0; i<v.size(); ++i) kK(x)[i]=kget(v[i]);
 return x;
}

K kget(const TensorVector& v,K x) { // x-nullptr by default, else indices
 if(!x) {
  K r=ktn(0,v.size());
  for(J i=0; i<r->n; ++i) kK(r)[i]=kget(v[i]);
  return razelist(r);
 } else if(x->t == -KJ) {
  return kget(v.at(x->j));
 } else if(x->t == KJ) {
  K r=ktn(0,x->n);
  for(J i=0; i<x->n; ++i) kK(r)[i]=kget(v.at(kJ(x)[i]));
  return razelist(r);
 } else {
  TORCH_ERROR("vector: expecting 2nd arg of long indices, given ",kname(x));
 }
}

K kget(const TensorDict& d,K x) { // x-nullptr by default, can contain sym(s) for indexing
 if(!x) {
  J i=0; K k=ktn(KS,d.size()),v=ktn(0,d.size());
  for(const auto &a:d) {
   kS(k)[i]=cs(a.key().c_str());
   kK(v)[i]=kget(a.value());
   ++i;
  }
  return xD(k,razelist(v));
 } else if(x->t == -KS) {
  return kget(d[x->s]);
 } else if(x->t == KS) {
  K r=ktn(0,x->n);
  for(J i=0; i<x->n; ++i)
   kK(r)[i]=kget(d[kS(x)[i]]);
  return razelist(r);
 } else {
  TORCH_ERROR("dict: expecting 2nd arg of symbol(s) for indexing, given ",kname(x));
 }
}

// ------------------------------------------------------------------------
// kget - return module input/output as array or list/dict of arrays
// kin - return module input as tensor,vector/dictionary of tensors
// kout - return module output as tensor or vector of tensors
// ------------------------------------------------------------------------
K kget(const Tuple& t) {
 return kget(TensorVector{std::get<0>(t),std::get<1>(t)});
}

K kget(const Nested& t) {
 return kget(TensorVector{std::get<0>(t),
                          std::get<0>(std::get<1>(t)),
                          std::get<1>(std::get<1>(t))});
}

K kget(const Input& x) {
 if       (auto a=c10::get_if<Tensor>(&x)) {       return kget(*a);
 } else if(auto a=c10::get_if<TensorVector>(&x)) { return kget(*a);
 } else if(auto a=c10::get_if<TensorDict>(&x))   { return kget(*a);
 } else if(       c10::get_if<Empty>(&x))        { return knull();
 } else { TORCH_ERROR("unrecognized input");
 }
}

K kget(const Output& x) {
 if       (auto a=c10::get_if<Tensor>(&x)) {       return kget(*a);
 } else if(auto a=c10::get_if<TensorVector>(&x)) { return kget(*a);
 } else if(auto a=c10::get_if<Tuple>(&x)) {        return kget(*a);
 } else if(auto a=c10::get_if<Nested>(&x)) {       return kget(*a);
 } else { TORCH_ERROR("unrecognized output");
 }
}

K kin(const Input& x) {
 if       (auto a=c10::get_if<Tensor>(&x)) {       return kten(*a);
 } else if(auto a=c10::get_if<TensorVector>(&x)) { return kvec(*a);
 } else if(auto a=c10::get_if<TensorDict>(&x))   { return kdict(*a);
 } else if(       c10::get_if<Empty>(&x))        { return knull();
 } else { TORCH_ERROR("unrecognized input");
 }
}

K kout(const Output& o) {
 if(auto a=c10::get_if<Tensor>(&o)) {
  return kten(*a);
 } else if(auto a=c10::get_if<TensorVector>(&o)) {
  return kvec(*a);
 } else if(auto a=c10::get_if<Tuple>(&o)) {
  return kvec({std::get<0>(*a),std::get<1>(*a)});
 } else if(auto a=c10::get_if<Nested>(&o)) {
  return kvec({std::get<0>(*a), std::get<0>(std::get<1>(*a)), std::get<1>(std::get<1>(*a))});
 } else {
  TORCH_ERROR("unrecognized output from forward calculation");
 }
}

// -------------------------------------------------------------------------------
// checkint - check options for modes not implemented for integral types
// checksparse - check options for sparse tensor, signal nyi combinations
// to - change tensor/vector device/type, create new tensor if copy flag set
// -------------------------------------------------------------------------------
static bool checkint(const TensorOptions& o,Tensormode m=Tensormode::undefined);
static bool checkint(const TensorOptions& o,Tensormode m) {
 if(o.has_dtype() && torch::isIntegralType(torch::typeMetaToScalarType(o.dtype()),true)) {
  switch(m) {
   case Tensormode::rand:
   case Tensormode::randn:
    TORCH_ERROR(modesym(m), ": not implemented for ",optdtype(o.dtype())," tensors");
   default: break;
  }
  return true;
 } else {
  return false;
 }
}

static bool checksparse(const TensorOptions& o,Tensormode m=Tensormode::undefined);
static bool checksparse(const TensorOptions& o,Tensormode m) {
 if(o.layout()==torch::kSparse || m==Tensormode::sparse) {
  TORCH_CHECK(!o.has_layout() || o.layout()==torch::kSparse, "tensor: sparse mode incompatible with layout set to ",optlayout(o.layout()));
  TORCH_CHECK(!o.pinned_memory(), "sparse tensors cannot have pinned memory");
  TORCH_CHECK(!(o.has_memory_format() && (o.memory_format_opt().value()==torch::MemoryFormat::ChannelsLast ||
                                          o.memory_format_opt().value()==torch::MemoryFormat::ChannelsLast3d)),
              "sparse tensors cannot use memory formats with channels as last dimension");
  switch(m) {
   case Tensormode::undefined:
   case Tensormode::complex:
   case Tensormode::empty:
   case Tensormode::sparse:
   case Tensormode::zeros:
    break;
   default: TORCH_ERROR(modesym(m), ": not implemented for sparse tensors");
  }
  return true;
 } else {
  return false;
 }
}

static Tensor to(const Tensor& t,const TensorOptions& o,bool a=false,bool b=false);
static Tensor to(const Tensor& t,const TensorOptions& o,bool a,bool b) {
 // as of version 1.8.0, errors using to() with sparse or gradients:
 // pinned memory doesn't seem to be handled from within any .to() method
 //"Operators taking TensorOptions cannot take a TensorOptions with options.requires_grad set as true. This isn't implemented yet."
 //"to(options) doesn't support converting to a different layout, but got self.layout being Strided and options.layout set as Sparse"
 if(!t.defined())
  return t;
 else if(checksparse(o))
   return t.to_sparse().to(o.requires_grad(false),a,b).set_requires_grad(o.requires_grad());
 else if(t.is_sparse() && o.has_layout() && o.layout()==torch::kStrided)
   return to(t.to_dense(),o);
 else if(o.pinned_memory())
  return t.to(o.requires_grad(false),a,b).pin_memory().set_requires_grad(o.requires_grad());
 else
  return t.to(o.requires_grad(false),a,b).set_requires_grad(o.requires_grad());
}

K to(Kten* t,const TensorOptions& o,bool a,bool b) {
 TORCH_CHECK(t->t.defined(),"to: cannot change attribute(s) of an undefined tensor");
 auto r=to(t->t,o,a,b);
 if(b)                 // if copy flag set
  return kten(r);      // return new tensor
 if(!t->t.is_same(r))  // else if device/dtype caused new tensor
  t->t=r;              // replace tensor in k ptr
 return (K)0;
}

void to(TensorDict& d,const TensorOptions& o,bool a) {
 for(auto& i:d) {
  if(i.value().defined()) {
   auto t=i.value().to(o,a);
   if(!i.value().is_same(t)) i.value()=std::move(t);
  }
 }
}

void to(TensorVector& v,const TensorOptions& o,bool a) {
 for(auto& t:v) {
  if(t.defined()) {
   auto r=t.to(o,a);
   if(!t.is_same(r)) t=std::move(r);
  }
 }
}

// ---------------------------------------------------------------------------------------
// kputscalar - copy single k value to CPU tensor scalar
// kdepth - check k array at depth for consistent datatype, size, etc, throw errors
// kputs - descend depth of k array, determining dim & sizes, copying data types to tensor
// kput - controlling function to read k array, create tensor and copy data at depth
// ---------------------------------------------------------------------------------------
void kputscalar(K x,Tensor &t) {
 Scalar s;
 TORCH_CHECK(xscalar(x,s), "unable to translate k ",kname(x->t)," to scalar tensor");
 t=torch::full({},s,maptype(x->t));
}

static void kdepth(K x,I i,H k,Ksize &s){
 if(x->t < 0) {
  TORCH_ERROR("unable to map mixed array to tensor: ",kname(x->t)," encountered at depth ",i);
 } else if(k != nh) {             // if base type already encountered
  I j=s.size()-1;                 // last size index
  if(x->n != s[i]) {              // check that dimensions are consistent
   TORCH_ERROR("dimension mismatch at depth ",i,", ",s[i]," vs ",x->n);
  } else if(x->t != (i<j ? 0 : k)) {  // check for same data type at same depth
   TORCH_ERROR("type mismatch at depth ",i,", ",kname(i<j ? 0 : k)," vs ",kname(x->t));
  }
 } else {
  s.push_back(x->n);              // no error, no base type yet, accumulate sizes
 }
}

static void kputs(K x,I i,H &k,Ksize &s,J &b,S &p,Tensor &t) {
 kdepth(x,i,k,s);
 if(x->t || !x->n) {     // if base data type or empty
  if(k==nh)  {           // if first encounter w'base data type
   k=x->t;
   t=k ? torch::empty(s, maptype(k)) : torch::empty(s);
   b=t.element_size() * s[i];  // bytes to copy
   p=(S)t.data_ptr();          // contiguous data pointer
  }
  memcpy(p,kG(x),b); p+=b;
 } else {
  for(I j=0;j<x->n;++j) kputs(kK(x)[j],i+1,k,s,b,p,t);
 }
}

Tensor kput(K x) {        
 H k=nh;                   // fill w'base data type for nested k value
 J b=0;                    // fill w'bytes to copy
 Ksize s;                  // fill w'k array size at each depth
 S p=nullptr;              // data pointer for created tensor
 Tensor t;                 // undefined tensor
 if(x->t < 0)              // if scalar
  kputscalar(x,t);         // create scalar backed by tensor
 else if(!xnull(x))        // else go through the depth of the array
  kputs(x,0,k,s,b,p,t);    // until base data type encountered
 return t;
}

Tensor kput(K x,J i) {
 if(xind(x,i)) 
  return kput(kK(x)[i]);
 else
  TORCH_ERROR("unable to index ",kname(x->t),", element: ",i);
}

// --------------------------------------------------------------------
// broadcast - true if tensor y can be broadcast to fill tensor x
// --------------------------------------------------------------------
bool broadcast(const Tensor& x,const Tensor& y) {
 if(x.dim()<y.dim())
  return false;
 int64_t j=x.dim()-y.dim();
 for(int64_t i=0; i<y.dim(); ++i)
  if(x.size(i+j) != y.size(i) && y.size(i) != 1)
   return false;
 return true;
}

// --------------------------------------------------------------------
// kput - given indices & values from k, [re]set TensorVector elements
// --------------------------------------------------------------------
static void kput(TensorVector& v,J i,const Tensor& t) {
 if(i == nj || i == -1)
  v.emplace_back(t);
 else
  v.at(i)=t;
}

static bool kput(TensorVector& v,J i,K x) {
 Tensor *t=nullptr;
 if(xptr(x)) {
  t=xten(x);
  TORCH_CHECK(t, "vector: not implemented for ",kname(x));
 }
 kput(v, i, t ? *t : kput(x));
 return t;
}

static void kput(TensorVector& v,K x,K y) {
 if(x->t == -KJ) {
  if(kput(v,x->j,y))
   kfree(y);
 } else if(x->t == KJ) {
  if(y->t) {
   Tensor t=kput(y);
   TORCH_CHECK(x->n == t.numel(), "vector: length error, index count of ", x->n, " with ", t.numel(), " value(s)");
   for(J i=0; i<x->n; ++i)
    kput(v,kJ(x)[i],t.dim() ? t[i].clone() : t);
  } else {
   TORCH_CHECK(x->n == y->n, "vector: length error, index count of ", x->n, " with ", y->n, " value(s)");
   bool b=false; TensorVector w;
   for(J i=0; i<x->n; ++i) if(kput(w, -1, kK(y)[i])) b=true; // put k values/tensor ptrs in temp vector
   for(J i=0; i<x->n; ++i) kput(v,kJ(x)[i],w[i]);            // if no error, add to existing vector
   if(b)                                                     // if any tensor pointers, free
    for(J i=0; i<y->n; ++i)
     if(xten(y,i)) kfree(y,i);
  }
 } else {
  TORCH_ERROR("vector: expecting long indices as 2nd arg, given ",kname(x));
 }
}

// ------------------------------------------------------------------------
// kput - put tensors/array in dictionary using symbols and arrays/tensors
// ------------------------------------------------------------------------
static void kput(Cast c,TensorDict& d,S s,const Tensor& t) {
 if(auto a=d.find(s)) {
  // attempt to update in place if gradient required or dictionary built as parameter/buffer list
  if(a->requires_grad() || c==Cast::parameter || c==Cast::buffer) {
   torch::NoGradGuard g;
   TORCH_CHECK(broadcast(*a,t), "dict[`",s,"] with size of ",a->sizes()," cannot be updated with new values of size ",t.sizes());
   a->copy_(t);
  } else {
   d[s]=std::move(t);
  }
 } else {
  d.insert(s,std::move(t));
 }
}

static bool kput(Cast c,TensorDict& d,S s,K x) {
 Tensor* t=nullptr;
 if(xptr(x)) {
  t=xten(x);
  TORCH_CHECK(t, "dict: not implemented for ",kname(x));
 }
 kput(c,d,s,t ? *t : kput(x));
 return t;
}

static void kput(Cast c,TensorDict& d,K x,K y) {
 if(x->t == -KS) {
  if(kput(c,d,x->s,y))
   kfree(y);
 } else if(x->t == KS) {
  if(y->t) {
   Tensor t=kput(y);
   TORCH_CHECK(x->n == t.numel(), "dict: length error, ", x->n, " key(s) with ", t.numel(), " value(s)");
   for(J i=0; i<x->n; ++i)
    kput(c,d,kS(x)[i],t.dim() ? t[i].clone() : t);
  } else {
   TORCH_CHECK(x->n == y->n, "dict: length error, ", x->n, " key(s) with ", y->n, " value(s)");
   bool b=false; TensorVector w;
   for(J i=0; i<x->n; ++i) if(kput(w, -1, kK(y)[i])) b=true;  // add tensors/arrays to temp vector
   for(J i=0; i<x->n; ++i) kput(c, d, kS(x)[i], w[i]);        // if no error, add to dictionary
   if(b)                                                      // if any tensor pointers, free
    for(J i=0; i<y->n; ++i)
     if(xten(y,i)) kfree(y,i);
  }
 } else {
  TORCH_ERROR("dict: given ptr, expecting symbol keys & values, but 2nd arg is ",kname(x));
 }
}

TensorDict kputd(K x) {
 TensorDict d;
 if(xdict(x) || (!x->t && x->n==2 && (kK(x)[0]->t==KS || kK(x)[0]->t==-KS)))
  kput(Cast::tensor,d,kK(x)[0],kK(x)[1]);
 else if(!xempty(x))
  TORCH_ERROR("dict: expecting k dictionary or (syms;vals), given ",kname(x));
 return d;
}
 
// --------------------------------------------------------------------------------------
// complextype - get component data type from complex data type or default data type
// complexdim  - determine first/last dim on which to split for real/imaginary
// --------------------------------------------------------------------------------------
static ScalarType complextype(c10::optional<TypeMeta> t,ScalarType a=ScalarType::Undefined, ScalarType b=ScalarType::Undefined);
static ScalarType complextype(c10::optional<TypeMeta> t,ScalarType a,ScalarType b) {
 ScalarType d;
 if(torch::isFloatingType(a) && torch::isFloatingType(b)) {
  TORCH_CHECK(a==b, "complex: real input is ",optdtype(a),", imaginary input is ",optdtype(b));
 }
 if(t) {
  switch(torch::typeMetaToScalarType(*t)) {
   case torch::kComplexHalf:   d=torch::kHalf; break;
   case torch::kComplexFloat:  d=torch::kFloat; break;
   case torch::kComplexDouble: d=torch::kDouble; break;
   default:
    TORCH_ERROR("unable to create complex tensor with given datatype: ",optdtype(*t));
  }
 } else if(torch::isFloatingType(a)) {
  d=a; 
 } else if(torch::isFloatingType(b)) {
  d=b; 
 } else {
  d=torch::get_default_dtype_as_scalartype();
  if(!isFloatingType(d)) d=torch::kFloat;
 }
 return d;
}

static int64_t complexdim(const Tensor& a,c10::optional<bool> b) {
 bool c=b ? *b : env().complexfirst; int64_t d=c ? 0 : -1;
 TORCH_CHECK(a.dim(),      "complex: single input array must have one or more dimensions");
 TORCH_CHECK(a.size(d)==2, "complex: single input array must have a ",
                           (c ? "first" : "last")," dimension of size 2 (real",(c ?  ";" : ",'"),
                           "imaginary), given size of ",a.sizes());
 return d;
}

// --------------------------------------------------------------------------------------
// complex1 - make a complex tensor from single input: (real,'imag) or (real;imag)
// --------------------------------------------------------------------------------------
static Tensor complex1(const Tensor& a,int64_t d,Tensor& r) {  // w'output tensor
 return torch::complex_out(r, a.select(d,0), a.select(d,1));
}

static Tensor complex1(const Tensor& a,Tensor& r,c10::optional<bool> b=c10::nullopt);
static Tensor complex1(const Tensor& a,Tensor& r,c10::optional<bool> b) {
 return complex1(a.to(complextype(r.dtype())), complexdim(a,b), r);
}

static Tensor complex1(const Tensor& a,int64_t d) {   // no output tensor
 return torch::complex(a.select(d,0), a.select(d,1));
}

static Tensor complex1(const Tensor& a,const TensorOptions& o,c10::optional<bool> b=c10::nullopt);
static Tensor complex1(const Tensor& a,const TensorOptions& o,c10::optional<bool> b) {
 return complex1(a.to(complextype(o.dtype_opt(),a.scalar_type())), complexdim(a,b)).to(o);
}

// --------------------------------------------------------------------------------------
// complex2 - make a complex tensor from real & imaginary inputs
// --------------------------------------------------------------------------------------
static Tensor complex2(const Tensor& a,const Tensor& b,Tensor& r) {  // w'output tensor
 auto t=complextype(r.dtype(), a.scalar_type(), b.scalar_type());
 return torch::complex_out(r,a.to(t),b.to(t));
}

static Tensor complex2(const Tensor& a,const Tensor& b,TensorOptions& o) {
 auto t=complextype(o.dtype_opt(), a.scalar_type(), b.scalar_type());
 return torch::complex(a.to(t), b.to(t));
}


// ----------------------------------------------------------------------------------------
// tensorlike - tensor creation routines, e.g. ones_like() where tensor given as template
// tensorout - tensor creation routines, e.g. ones_out(), where output tensor is given
// tensoropt - tensor creation routines where tensor size and option(s) given
// tensormode - determines whether a template tensor or output tensor given w'other args
// tensorput - put k value(s) -> tensor, return new tensor ptr unless output tensor given
// tensorget - given tensor ptr and optional flag & indexing args, get result for k array
// vectorptr - given vector ptr, return tensor pointers, or single pointer if index given
// dictptr - given dictionary ptr, return dictionary of tensor pointers, or single pointer
// tensor - high level function to create/retrieve/move/recast tensor from k
// ----------------------------------------------------------------------------------------
static void tensorlike(K x,Tensormode m,const Tensor &t,Tensor &r) {  // t:input, r:result tensor
 J i,j; Scalar s; TensorOptions o;
 bool b=xopt(x,x->n-1,o); I nx=x->n-b;  //set flag if options given, count non-option args
 switch(m) {
  case Tensormode::empty: if(nx==2) r=torch::empty_like(t,o); break;
  case Tensormode::zeros: if(nx==2) r=torch::zeros_like(t,o); break;
  case Tensormode::ones:  if(nx==2) r=torch::ones_like(t,o);  break;
  case Tensormode::rand:  if(nx==2) r=torch::rand_like(t,o);  break;
  case Tensormode::randn: if(nx==2) r=torch::randn_like(t,o); break;
  case Tensormode::full:
   if(nx==3 && xscalar(x,2,s))
    r=torch::full_like(t,s,o.has_dtype() ? o : o.dtype(maptype(kK(x)[2]->t)));
   break;
  case Tensormode::randint:
   if     (nx==3 && xlong(x,2,j))                 r=torch::randint_like(t,j,o);
   else if(nx==4 && xlong(x,2,i) && xlong(x,3,j)) r=torch::randint_like(t,i,j,o);
   break;
  default:
   TORCH_ERROR("tensor: mode `",kK(x)[0]->s," not implemented with input tensors");
   break;
 }
}

static void tensorout(K x,Tensormode m,Tensor &t,Tensor &r) {  // t:output, r:result tensor
 double e; J i,j; Scalar a,z,n; IntArrayRef s;
 bool b=xsize(x,1,s);  //true if size is given as 2nd arg (last arg is output tensor)
 switch(m) {
  case Tensormode::empty: if(b && x->n==3) r=torch::empty_out(t,s); break;
  case Tensormode::zeros: if(b && x->n==3) r=torch::zeros_out(t,s); break;
  case Tensormode::ones:  if(b && x->n==3) r=torch::ones_out(t,s); break;
  case Tensormode::rand:  if(b && x->n==3) r=torch::rand_out(t,s); break;
  case Tensormode::randn: if(b && x->n==3) r=torch::randn_out(t,s); break;
  case Tensormode::full:  if(b && x->n==4 && xscalar(x,2,a)) r=torch::full_out(t,s,a); break;
  case Tensormode::randperm: if (x->n==3 && xlong(x,1,i)) r=torch::randperm_out(t,i); break;
  case Tensormode::randint:
   b=xsize(x,x->n-2,s);
   if     (b && x->n==4 && xlong(x,1,j))                 r=torch::randint_out(t,j,s);
   else if(b && x->n==5 && xlong(x,1,i) && xlong(x,2,j)) r=torch::randint_out(t,i,j,s);
   break;
  case Tensormode::eye:
    if     (x->n==3 && xlong(x,1,i))                 r=torch::eye_out(t,i);
    else if(x->n==4 && xlong(x,1,i) && xlong(x,2,j)) r=torch::eye_out(t,i,j);
    break;
  case Tensormode::range:
  case Tensormode::arange:
   b=m==Tensormode::range;
   if     (x->n==3 && xnum(x,1,z))                              r = b ? torch::range_out(t,0,z)   : torch::arange_out(t,z);
   else if(x->n==4 && xnum(x,1,a) && xnum(x,2,z))               r = b ? torch::range_out(t,a,z)   : torch::arange_out(t,a,z,1);
   else if(x->n==5 && xnum(x,1,a) && xnum(x,2,z) && xnum(x,3,n))r = b ? torch::range_out(t,a,z,n) : torch::arange_out(t,a,z,n);
   break;
  case Tensormode::linspace:
  case Tensormode::logspace:
   b=m==Tensormode::logspace; i=100; e=10.0; //default of 100 steps, base 10
   if(xnum(x,1,a) && xnum(x,2,z) && (x->n==4 || (xlong(x,3,i) && (x->n==5 || (x->n==6 && b && xnum(x,4,e))))))
    r = b ? torch::logspace_out(t,a,z,i,e) : torch::linspace_out(t,a,z,i);
   break;
  case Tensormode::complex:
   if(x->n==3) {
    r=complex1(kput(x,1),t);
   } else if(x->n==4) {
    if(xbool(x,2,b))
     r=complex1(kput(x,1),b,t);
    else
     r=complex2(kput(x,1),kput(x,2),t);
   }
   break;
  case Tensormode::sparse:
   TORCH_ERROR("tensor: sparse not implemented with output tensors");
   break;
  default:
   TORCH_ERROR("tensor: unexpected tensor mode `",kK(x)[0]->s," supplied with output tensor");
   break;
 }
}

static void tensoropt(K x,Tensormode m,Tensor &r) {
 double e; J i,j,nx=x->n; Scalar a,z,n; IntArrayRef s; TensorOptions o;
 bool b=xopt(x,x->n-1,o); if(b) nx--;                         //track if options in last arg
 bool sz=xsize(x,1,s) && nx==((m==Tensormode::full) ? 3 : 2); //2nd arg is size & correct arg count
 checksparse(o,m); checkint(o,m);
 switch(m) {
  case Tensormode::empty: if(sz) r=torch::empty(s,o); break;
  case Tensormode::zeros: if(sz) r=torch::zeros(s,o); break;
  case Tensormode::ones:  if(sz) r=torch::ones(s,o); break;
  case Tensormode::rand:  if(sz) r=torch::rand(s,o); break;
  case Tensormode::randn: if(sz) r=torch::randn(s,o); break;
  case Tensormode::full:
   if(sz && xscalar(x,2,a))
    r=torch::full(s,a,o.has_dtype() ? o : o.dtype(maptype(kK(x)[2]->t)));
   break;
  case Tensormode::randperm:
   if(!o.has_dtype()) o=o.dtype(torch::kLong);
   if(nx==2 && xlong(x,1,i)) r = torch::randperm(i,o);
   break;
  case Tensormode::randint:
   sz=xsize(x,nx-1,s); // true if size is supplied as last non-options arg
   if(!o.has_dtype()) o=o.dtype(torch::kLong);
   if     (sz && nx==3 && xlong(x,1,j))                 r=torch::randint(j,s,o);
   else if(sz && nx==4 && xlong(x,1,i) && xlong(x,2,j)) r=torch::randint(i,j,s,o);
   break;
  case Tensormode::eye:
    if     (nx==2 && xlong(x,1,i))                 r=torch::eye(i,o);
    else if(nx==3 && xlong(x,1,i) && xlong(x,2,j)) r=torch::eye(i,j,o);
    break;
  case Tensormode::range:
   if     (nx==3 && xnum(x,1,a) && xnum(x,2,z))               r=torch::range(a,z,o);
   else if(nx==4 && xnum(x,1,a) && xnum(x,2,z) && xnum(x,3,n))r=torch::range(a,z,n,o);
   break;
  case Tensormode::arange:
   b=!o.has_dtype();
   if(nx==2 && xnum(x,1,z)) {
    if(b && z.isIntegral(false)) o=o.dtype(torch::kLong);
    r=torch::arange(z,o);
   } else if(nx==3 && xnum(x,1,a) && xnum(x,2,z)) {
    if(b && a.isIntegral(false) && z.isIntegral(false)) o=o.dtype(torch::kLong);
    r=torch::arange(a,z,o);
   } else if(nx==4 && xnum(x,1,a) && xnum(x,2,z) && xnum(x,3,n)) {
    if(b && a.isIntegral(false) && z.isIntegral(false) && n.isIntegral(false)) o=o.dtype(torch::kLong);
    r=torch::arange(a,z,n,o);
   }
   break;
  case Tensormode::linspace:
  case Tensormode::logspace:
   b=m==Tensormode::logspace; i=100; e=10.0; //default of 100 steps, base 10
   if(xnum(x,1,a) && xnum(x,2,z) && (nx==3 || (xlong(x,3,i) && (nx==4 || (nx==5 && b && xnum(x,4,e))))))
    r = b ? torch::logspace(a,z,i,e,o) : torch::linspace(a,z,i,o);
   break;
  case Tensormode::complex:
   if(nx==2) {
    r=complex1(kput(x,1), o);
   } else if(nx==3) {
    if(xbool(x,2,b))
     r=complex1(kput(x,1), o, b);
    else
     r=complex2(kput(x,1), kput(x,2), o);
   }
   break;
  case Tensormode::sparse:
   // as of version 1.8.1, sparse_coo_tensor seems to ignore tensor options if indices & values supplied
   // there is a check if explicit strided setting and gradient required throws nyi, requiring workaround below
   // https://github.com/pytorch/pytorch/issues/55453
   checksparse(o,m);
   if(nx==2 && xsize(x,1,s))
    r=torch::sparse_coo_tensor(s,o);
   else if(nx==3)
    r=to(torch::sparse_coo_tensor(kput(x,1),kput(x,2),o.requires_grad(false)), o);
   else if(nx==4 && xsize(x,3,s))
    r=to(torch::sparse_coo_tensor(kput(x,1),kput(x,2),s,o.requires_grad(false)), o);
   break;
  default: break;
 }
 // most tensor creation functions don't support newer memory format options yet (as of version 1.8.1)
 if(o.has_memory_format() && r.suggest_memory_format() != o.memory_format_opt().value()) {
  torch::NoGradGuard g;
  r=r.is_pinned() ? r.contiguous(*o.memory_format_opt()).pin_memory() : r.contiguous(*o.memory_format_opt());
  if(o.requires_grad()) r.set_requires_grad(true);
 }
}

static K tensormode(K x,S s,Tensormode m) {
 Tensor t,r; bool in=false,out=false;
 if((in=xten(x,1,t)))            tensorlike(x,m,t,r); // input tensor is 2nd arg
 else if((out=xten(x,x->n-1,t))) tensorout(x,m,t,r);  // output tensor is final arg
 else                            tensoropt(x,m,r);    // no input/output tensor
 TORCH_CHECK(r.defined(),"unrecognized argument(s) for tensor creation mode: ",s);
 return out ? (K)0 : kten(r);
}

static K tensorput(K x) {
 Tensor r,t; TensorOptions o;
 t=((xopt(x,1,o) || xten(x,1,r)) && x->n==2) ? kput(x,0) : kput(x);
 if(r.defined()) {
  r.resize_(t.sizes()).copy_(t,true);
  return (K)0;
 } else {
  //return kten(to(t,o));  // NULL TENSOR
  return kten(t.defined() ? to(t,o) : t);
 }
}

static Tensor tensorget(const Tensor& t,J d,K x) {   // d:dimension, x:index/indices
 if(x->t == -KJ)
  return t.select(d,x->j);
 else if(x->t == KJ)
  return torch::index_select(t,d,kput(x).to(t.device()));
 else
  TORCH_ERROR("tensor: last arg expected to be long(s) for indexing, given ",kname(x));
}

Tensor tensorget(const Tensor& t,K x) {
 bool b=false,c; J d=0; Tensor r;
 if((c=xbool(x,1,b)))
  TORCH_CHECK(t.is_complex(), "tensor: optional flag is only for complex tensors");
 if(x->n == 1+c) {                     // ptr or (ptr;flag)
  r=t;
 } else if(x->n == 2+c) {              // (ptr;ind) or (ptr;flag;ind)
  r=tensorget(t,d,kK(x)[x->n-1]);
 } else if(x->n == 3+c) {              // (ptr;dim;ind) or (ptr;flag;dim;ind)
  TORCH_CHECK(xlong(x,1+c,d), "tensor: ",(c ? "2nd" : "3rd")," arg of dimension expected as a long scalar, given ",kname(x,1+c));
  r=tensorget(t,d,kK(x)[x->n-1]);
 } else {
  TORCH_ERROR("tensor: up to ",3+c," args expected, (tensor;", (c ? "flag;" : ""),"dim;ind), but ",x->n," args given");
 }
 return c ? toreal(r,b) : r;
}

static K vectorptr(const TensorVector& v,K x) {
 if(x->n==1) {                                 // no additional args, return list of tensor ptrs
  J i=0; K r=ktn(0,v.size());
  for(const auto& t:v) kK(r)[i++]=kten(t);
  return r;
 } else if(x->n==2) {                          // 2nd arg of single index or list of indices
  K y=kK(x)[1];
  if(y->t == -KJ) {
   return kten(v.at(y->j));                              // single index, return tensor ptr
  } else if(y->t == KJ) {                                // indices, return list of selected tensor ptrs
   K r=ktn(0,y->n);
   for(J i=0; i<y->n;++i) kK(r)[i]=kten(v.at(kJ(y)[i]));
   return r;
  } else {
   TORCH_ERROR("tensor: given vector, 2nd arg expected to be long(s) for indexing, not ",kname(y));
  }
 } else {
  TORCH_ERROR("tensor: given vector, expecting no more than one additional indexing argument but given ",x->n-1," additional args");
 }
}

static K dictptr(const TensorDict& d,K x) {
 if(x->n==1) {                                       // no additional args, return k dict of tensor ptrs
  J i=0; K k=ktn(KS,d.size()),v=ktn(0,d.size());
  for(const auto &a:d) {
   kS(k)[i]=cs(a.key().c_str());
   kK(v)[i]=kten(a.value());
   ++i;
  }
  return xD(k,v);
 } else if(x->n==2) {                                 // additional indexing arg
  K y=kK(x)[1];
  if(y->t == -KS) {                                   // single symbol, return tensor pointer
   return kten(d[y->s]);
  } else if(y->t == KS) {                             // list of symbols, return k dict of selected tensor ptrs
    K r=ktn(0,y->n);
    for(J i=0; i<y->n;++i) kK(r)[i]=kten(d[kS(y)[i]]);
    return r;
  } else {
   TORCH_ERROR("tensor: given dictionary, 2nd arg expected to be symbols(s) for indexing, not ",kname(y));
  }
 } else {
  TORCH_ERROR("tensor: given dictionary, expecting no more than one additional indexing argument but given ",x->n-1," additional args");
 }
}

KAPI tensor(K x) {
 KTRY
  S s; Tensormode m; Ktag *g;
  if((g=xtag(x)) || (g=xtag(x,0))) {
   switch(g->a) {
    case Class::tensor: return kget(tensorget(g->tensor(),x));
    case Class::vector: return vectorptr(g->vector(), x);
    case Class::dict:   return dictptr(g->dict(), x);
    default: TORCH_ERROR("tensor not implemented for ",mapclass(g->a));
   }
  } else if(xmode(x,0,s,m)) {
   return tensormode(x,s,m);
  } else {
   return tensorput(x);
  }
 KCATCH("tensor");
}

// ------------------------------------------------------------------------------------------
// tensor vector fns: 
// ------------------------------------------------------------------------------------------
// vec - initialize vector of tensors from k array, tensor ptr(s) or some mix of both
// vector - create vector of tensors, or return vector or vector element, or replace element
// dict - create dictionary of tensors, or return dictionary value(s)
// ------------------------------------------------------------------------------------------
TensorVector vec(K x,bool b) {   // b: true if any encountered tensor ptr to be de-referenced
 TensorVector v;
 if(x->t) {
  Tensor t=kput(x);
  if(t.dim())
   for(int64_t i=0;i<t.size(0);++i)
    v.emplace_back(t[i].clone());
  else
   v.emplace_back(t);
 } else if(xptr(x)) {
  if(kput(v,-1,x))
   if(b) kfree(x);
 } else {
  bool a=false;
  for(J i=0;i<x->n;++i)
   if(kput(v, -1, kK(x)[i])) a=true;
  if(a && b)
   for(J i=0;i<x->n;++i) if(xptr(x,i)) kfree(x,i);
 }
 return v;
}

KAPI vector(K x) {
 KTRY
  if(auto* v=xvec(x)) {             // if previously created vector, return as k list
   return kget(*v);
  } else if(auto* v=xvec(x,0)) {                 // if previously created vector
   if(x->n==2) {                                 // 2 args
    if(auto *w=xvec(x,1)) {                      // add additional vector
     for(size_t i=0,n=w->size(); i<n; ++i)
      v->emplace_back(w->at(i));         // add via index in case vector added to self
     if(v!=w) kfree(x,1);                // free vector added unless same
     return (K)0;
    } else {                            // else index into vector via 2nd arg of index/indices
     return kget(*v,kK(x)[1]);
    }
   } else if(x->n==3) {                 // if indices and values supplied
    kput(*v, kK(x)[1], kK(x)[2]);
    return (K)0;
   } else {
    TORCH_ERROR("vector: given ptr, expecting indices, vector ptr or (indices;values), but given ",x->n-1," additional arg(s)");
   }
  } else {
   return kvec(vec(x,true));
  }
 KCATCH("vector");
}

KAPI dict(K x) {
 KTRY
  TORCH_CHECK(x->t==0 || x->t==99, "dict: not implemented for ",kname(x));
  Ktag *g=xtag(x); if(!g) g=xtag(x,0); Cast c=g ? g->c : Cast::undefined;
  if(g && g->a == Class::dict) {
   TensorDict& d=g->dict();
   if(x->n==1) {                           // dict ptr is only argument
    return kget(d);                        // return dictionary of syms!values to k
   } else if(x->n==2) {
    if(xdict(x,1))                          // dict(ptr;kdict)
     return kput(c,d,kK(kK(x)[1])[0],kK(kK(x)[1])[1]), (K)0;
    else                                    // dict(ptr;sym(s))
     return kget(d,kK(x)[1]);
   } else if(x->n==3) { 
    return kput(c,d,kK(x)[1],kK(x)[2]), (K)0;
   } else {
    TORCH_ERROR("dict: expecting 1-3 args, but ",x->n," args supplied, not one of ptr, (ptr;dict), (ptr;syms), (ptr;syms;vals)");
   }
  } else {
   return kdict(kputd(x));
  }
 KCATCH("dict");
}

// --------------------------------------------------------------------------------------
//  complex tensors
// --------------------------------------------------------------------------------------
// kreal - handle api calls to extract real & imaginary parts of tensor
// real,imag - return real & imaginary parts of tensor as tensor or k value
// isreal - return boolean with 1's where value is real
// --------------------------------------------------------------------------------------
static K kreal(K x,Tensor(f)(const Tensor&),const char* nm) {
 KTRY
  bool b=false; Tensor *t=xten(x);
  if(!t) b=true, t=xten(x,0);  // enlisted tensor, return as k array
  TORCH_CHECK(t && t->is_complex() && x->n==1, nm, ": expects a complex tensor (enlist to return tensor ptr), given ",kname(x));
  // as of version 1.8.1, complex sparse tensors must  be made sparse real -> dense -> back to complex
  return kresult(b,f(t->is_sparse() ? torch::view_as_complex(sparsereal(*t).to_dense()) : *t));
 KCATCH(nm);
}

KAPI   real(K x) {return kreal(x, torch::real,   "real");}
KAPI   imag(K x) {return kreal(x, torch::imag,   "imag");}
KAPI isreal(K x) {return kreal(x, torch::isreal, "isreal");}

// ------------------------------------------------------------------------------------------
//  sparse tensors
// ------------------------------------------------------------------------------------------
// getsparse - handle api calls to extract indices & values from sparse tensor
// coalesce - colaesce a sparse tensor
// dense - return dense tensor given sparse tensor
// sparse - return sparse tensor given array, tensor, (array/tensor;dim), (array/tensor;mask)
// sparseindex - return non-zero indices of input as matrix/tensor, allow sparse dimension
// ------------------------------------------------------------------------------------------
static K getsparse(K x,bool i,const char* nm) {
 KTRY
  bool b=false; Tensor *t=xten(x);
  if(!t) b=true, t=xten(x,0);  // enlisted tensor, return as k array
  TORCH_CHECK(t && t->is_sparse() && x->n==1, nm, ": expects a sparse tensor (enlist to return tensor ptr), given ",kname(x));
  return kresult(b, i ? t->_indices() : t->_values());
 KCATCH(nm);
}

KAPI indices(K x) {return getsparse(x, true,  "indices");}
KAPI  values(K x) {return getsparse(x, false, "values");}

KAPI coalesce(K x) {
 KTRY
  Tensor *t=xten(x);
  TORCH_CHECK(t && t->is_sparse(), "coalesce: expecting sparse tensor, given ",kname(x));
  if(!t->is_coalesced())
   xtag(x)->set(t->coalesce());
  return (K)0;
 KCATCH("coalesce");
}

KAPI dense(K x) {
 KTRY
  Tensor *t=xten(x);
  TORCH_CHECK(t && t->is_sparse(), "dense: expecting sparse tensor, given ",kname(x));
  return kten(t->to_dense());
 KCATCH("dense");
}

KAPI sparse(K x) {
 KTRY
  int64_t d; Tensor *i=nullptr,*t=xten(x);
  if(t) {
   return kten(t->to_sparse());
  } else if (!x->t && x->n==2 && (xint64(x,1,d) || ((i=xten(x,1))))) {
   t=xten(x,0);
   if(i) {
    TORCH_CHECK(i->is_sparse(), "sparse: 2nd arg is not a sparse tensor, unable to use its indices");
    return kten((t ? *t : kput(x,0)).sparse_mask(*i));
   } else {
    return kten((t ? *t : kput(x,0)).to_sparse(d));
   }
  } else {
   TORCH_CHECK(xarray(x,2), "sparse: unrecognized arg(s), expecting input, (input;sparsedim) or (input;sparse tensor)");
   return kten(kput(x).to_sparse());
  }
 KCATCH("sparse");
}

KAPI sparseindex(K x) {
 KTRY
  int64_t d; Tensor *t;
  if((t=xten(x))) {
   return kten(t->nonzero().t());
  } else if(x->n==2 && xint64(x,1,d)) {
   t=xten(x,0);
   return kresult(t,std::get<0>(torch::unique_consecutive((t ? *t : kput(x,0)).nonzero().index_select(1,torch::arange(d)),false,false,0)).t());
  } else {
   return kget(kput(x).nonzero().t());
  }
 KCATCH("sparseindex");
}

// ----------------------------------------------------------------------------------------------
// kcat1 - check for tensor(s) at first level of a general list, e.g. cat(a;b)
// kcat2 - check for (inputs;dim) or (inputs;dim;out tensor) or (inputs;out tensor)
// kcat - handle args for cat/stack
// cat - join arrays or tensors along given dimension (dim 0 if none given)
// stack - join same-sized arrays along a new dimension (leading dim if none given)
// ----------------------------------------------------------------------------------------------
bool kcat1(K x) {for(J i=0;i<x->n;++i) if(xten(x,i)) return true; return false;}

bool kcat2(K x,int64_t& d, Tensor& r) {
 if(!x->t && x->n>1 && !kK(x)[0]->t)
  return (xint64(x,1,d) && (x->n==2 || (x->n==3 && xten(x,2,r)))) || (x->n==2 && xten(x,1,r));
 else
  return false;
}

K kcat(K x,bool b,const char* s) {
 KTRY
  TORCH_CHECK(!x->t, s," not implemented for ",kname(x->t));
  int64_t d=0; Tensor r; TensorVector *v;
  if((v=xvec(x))){
   return kten(b ? torch::cat(*v,d) : torch::stack(*v,d));
  } else if((v=xvec(x,0))) {
   TORCH_CHECK(kcat2(x,d,r), s," expects vector, (vector;dim), (vector;dim;out tensor) or (vector;out tensor)");
   if(r.defined()) 
    return (b ? torch::cat_out(r,*v,d) : torch::stack_out(r,*v,d)), (K)0;
   else
    return kten(b ? torch::cat(*v,d) : torch::stack(*v,d));
  } else if(kcat1(x)) {
   return kten(b ? torch::cat(vec(x,false),d) : torch::stack(vec(x,false),d));
  } else if((x->n==1 && !kK(x)[0]->t) || kcat2(x,d,r)) {
   K y=kK(x)[0];
   if(r.defined())
    return (b ? torch::cat_out(r,vec(y,false),d) : torch::stack_out(r,vec(y,false),d)), (K)0;
   else
    return kresult(!y->t && kcat1(y),b ? torch::cat(vec(y,false),d) : torch::stack(vec(y,false),d));
  } else {
   return kget(b ? torch::cat(vec(x,false),d) : torch::stack(vec(x,false),d));
  }
 KCATCH(s);
}

// as of v1.13.0, unable to use fn prototypes: cat uses ItensorRef, stack still uses TensorList
KAPI cat(K x)   {return kcat(x, true,  "cat");}
KAPI stack(K x) {return kcat(x, false, "stack");}

// ----------------------------------------------------------------------------------------------
// permute - k api for permute (not just for complex tensors)
// expand - expand tensor or array given sizes or tensor w'size to copy
// squeeze/unsqueeze - remove or add dimension to input array/tensor, boolean in-place option
// ----------------------------------------------------------------------------------------------
KAPI permute(K x) {
 KTRY
  Tensor *t=xten(x,0); IntArrayRef n;
  TORCH_CHECK(!x->t && x->n==2, "permute: unexpected arg(s), expecting (input/tensor; reordered dimensions)");
  TORCH_CHECK(xsize(x,1,n), "permute: expecting 2nd arg of reordered dimensions, given ",kname(x,1));
  return kresult(t, (t ? *t : kput(x,0)).permute(n));
 KCATCH("permute");
}

KAPI expand(K x) {
 KTRY
  IntArrayRef n; Tensor *t=xten(x,0),*s=xten(x,1);
  if((s || xsize(x,1,n)) && x->n==2) {
   if(t)
    return kten(s ? t->expand_as(*s) : t->expand(n));
   else
    return kget(s ? kput(x,0).expand_as(*s) : kput(x,0).expand(n));
  } else if(xsize(x,n) && n.size()==2) {
    return kget(torch::scalar_to_tensor(n[0]).expand(n[1]));
  } else {
   TORCH_ERROR("expand expects (input array/tensor; size) or (input array/tensor; tensor w'size to match)");
  }
 KCATCH("expand");
}

K ksqueeze(K x,bool a,const char* s) {
 KTRY
  bool b=false; int64_t d=nj; Tensor *t;
  if((t=xten(x))) {
   if(a) return kten(t->squeeze());
   else  return KERR("unsqueeze requires 2nd arg specifying dimension to add");
  } else if((xint64(x,1,d) && (x->n==2 || (x->n==3 && xbool(x,2,b)))) || (xbool(x,1,b) && x->n==2)) {
   TORCH_CHECK(a || d != nj, s," requires 2nd arg specifying dimension to add");
   if((t=xten(x,0))) {
    if(!a)           return b ? t->unsqueeze_(d),(K)0 : kten(t->unsqueeze(d));
    else if(null(d)) return b ? t->squeeze_(),   (K)0 : kten(t->squeeze());
    else             return b ? t->squeeze_(d),  (K)0 : kten(t->squeeze(d));
   } else {
    if(!a)           return kget(kput(x,0).unsqueeze(d));
    else if(null(d)) return kget(kput(x,0).squeeze());
    else             return kget(kput(x,0).squeeze(d));
   }
  } else if(a) {
   return kget(kput(x).squeeze());
  } else {
   TORCH_ERROR(s, ": unexpected arg(s), expects (input;dim;optional in-place flag)");
  }
 KCATCH(s);
}

KAPI squeeze(K x)   {return ksqueeze(x, true,  "squeeze");}
KAPI unsqueeze(K x) {return ksqueeze(x, false, "unsqueeze");}

// ---------------------------------------------------------------
// kindex - implementation of PyTorch select/index_select methods
// ---------------------------------------------------------------
KAPI kindex(K x) {
 KTRY
  int64_t d;
  TORCH_CHECK(!x->t, "index: not implemented for ",kname(x));
  TORCH_CHECK(x->n==3, "index: expecting 3 args, (input;dim;ind), given ",x->n);
  TORCH_CHECK(xint64(x,1,d), "index: 2nd argument of dimension expected, given ",kname(x,1));
  K y=kK(x)[2]; Tensor *t=xten(x,0), *j=xten(y);
  TORCH_CHECK(y->t==-KJ || y->t==KJ || j, "index: 3rd argument of index/indices expected, given ",kname(y));
  if(y->t==-KJ)
   return kresult(t, (t ? *t : kput(x,0)).select(d,y->j));
  else
   return kresult(t, (t ? *t : kput(x,0)).index_select(d,j ? *j : kput(y)));
 KCATCH("index");
}

// ----------------------------------------------------------------------------------------------
// tensorlong - tensor/vector attributes returned to k as long scalar
// tensorsym - tensor/vector attributes returned to k as a symbol, e.g. device
// tensorflag - tensor/vector attributes returned to k as a boolean, e.g. leaf
// tensorsize - tensor/vector attributes returned to k as a long list, e.g. size/stride
// tensorattr - handle tensor attribute queries according to k datatype returned
// ----------------------------------------------------------------------------------------------
static J storlong(const Storage& s,Attr a) {
 switch(a) {
  case Attr::ptr:         return (intptr_t)s.data();
  case Attr::ref:         return s.use_count();
  default: TORCH_ERROR(mapattr(a),": not implemented for storage");
 }
}

J tensorlong(const Tensor& t,Attr a) {
 switch(a) {
  case Attr::dim:       return t.dim();
  case Attr::itemsize:  return t.is_sparse() ? tensorlong(t._values(),a) : t.dtype().itemsize();
  case Attr::numel:     return t.is_sparse() ? t._values().numel()       : t.numel();
  case Attr::elements:  return objnum(t);
  case Attr::bytes:     return objbytes(t);
  case Attr::offset:    return t.is_sparse() ? nj : t.storage_offset();
  case Attr::ref:       return t.use_count();
  case Attr::sref:      return t.defined() ? t.storage().use_count() : 0;
  case Attr::weakref:   return t.defined() ? t.weak_use_count() : 0;
  case Attr::ptr:       return (intptr_t)t.unsafeGetTensorImpl();
  case Attr::sptr:      return !t.defined() || t.is_sparse() ? 0 : (intptr_t)t.storage().data();
  case Attr::densedim:  return t.is_sparse() ? t.dense_dim()  : t.dim();
  case Attr::sparsedim: return t.is_sparse() ? t.sparse_dim() : 0;
  case Attr::nnz:       return t.is_sparse() ? t._nnz() : (t.defined() ? t.count_nonzero().item().toLong() : 0);
  default: TORCH_ERROR(mapattr(a),": not implemented for tensors");
 }
}

S tensorsym(const Tensor& t,Attr a) {
 switch(a) {
  case Attr::device:   return t.defined() ? optdev(t.device()) : nullsym();
  case Attr::dtype:    return t.defined() ? optdtype(t.dtype()) : nullsym();
  case Attr::layout:   return t.defined() ? optlayout(t.layout()) : nullsym();
  case Attr::gradient: return t.defined() ? optgrad(t.requires_grad()) : nullsym();
  case Attr::gradfn:   return (S)(t.defined() && t.grad_fn() ?  cs(t.grad_fn()->name().c_str()) : "");
  case Attr::pinned:   return optpin(!t.defined() || t.is_sparse() ? false : t.is_pinned());
  case Attr::memory:   return t.defined() ? optmemory(t.suggest_memory_format()) : nullsym();
  default: TORCH_ERROR(mapattr(a),": not implemented for tensors");
 }
}

static bool tensorflag(const Tensor &t,Attr a) {
 switch(a) {
  case Attr::coalesced:    return t.is_sparse() ? t.is_coalesced() : true;
  case Attr::contiguous:   return t.is_sparse() ? false : t.is_contiguous();
  case Attr::contiguous2d: return t.is_sparse() ? false : t.is_contiguous(torch::MemoryFormat::ChannelsLast);
  case Attr::contiguous3d: return t.is_sparse() ? false : t.is_contiguous(torch::MemoryFormat::ChannelsLast3d);
  case Attr::defined:      return t.defined();
  case Attr::gradflag:     return t.requires_grad();
  case Attr::leaf:         return !t.defined() || t.is_leaf();
  case Attr::pinned:       return !t.defined() || t.is_sparse() ? false : t.is_pinned();
  case Attr::sparseflag:   return t.is_sparse();
  default: TORCH_ERROR(mapattr(a),": not implemented for tensors");
 }
}

static char tensorchar(const Tensor &t,Attr a) {
 switch(a) {
  case Attr::ktype:
   if(t.defined()) {
    for(const auto& a:env().dtype)
     if(t.dtype()==std::get<1>(a))
      return std::get<3>(a);
    TORCH_ERROR(mapattr(a),": unable to map PyTorch type ",t.dtype()," to k type");
   }
   return ' ';
  default: TORCH_ERROR(mapattr(a),": not implemented for tensors");
 }
}

K tensorsize(const Tensor &t,Attr a) {
 switch(a) {
  case Attr::size:    return klist(t.dim(),t.sizes().data());
  case Attr::stride:  return !t.defined() || t.is_sparse() ? ktn(0,0) : klist(t.dim(),t.strides().data());
  default: TORCH_ERROR(mapattr(a),": not implemented for tensors");
 }
}

K tensorattr(const Tensor &t,Ktype k,Attr a) {
 switch(k) {
  case -KJ: return kj(tensorlong(t,a));
  case  KJ: return tensorsize(t,a);
  case -KS: return ks(tensorsym(t,a));
  case -KB: return kb(tensorflag(t,a));
  case -KC: return kc(tensorchar(t,a));
  default: TORCH_ERROR(mapattr(a),": not implemented for tensors");
 }
}

// ----------------------------------------------------------------------------------------------
// vattr - handle vector/dictionary values or other iterable to extract attributes -> k list
// vectorattr - handle tensor vector attribute queries according to k datatype returned
// dictattr - handle tensor dictionary queries, return dictionary of attribute values
// ----------------------------------------------------------------------------------------------
template<typename V> static K vattr(const V &v,Ktype k,Attr a) {
 size_t i=0; K x=ktn(k<0 ? abs(k) : 0, v.size());
 try {
  for(const auto& t:v) {
   switch(k) {
    case -KJ: kJ(x)[i]=tensorlong(t,a); break;
    case  KJ: kK(x)[i]=tensorsize(t,a); break;
    case -KS: kS(x)[i]=tensorsym(t,a);  break;
    case -KB: kG(x)[i]=tensorflag(t,a); break;
    case -KC: kC(x)[i]=tensorchar(t,a); break;
    default: TORCH_ERROR(mapattr(a),": not implemented for tensor dictionary/vector");
   }
   ++i;
  }
 } catch(...) {
  if(x) r0(x);
  throw;
 }
 return x;
}

K vectorattr(const TensorVector &v,Ktype k,Attr a) {
 return vattr(v,k,a);
}

K dictattr(const TensorDict& d,Ktype k,Attr a) {
 K y=vattr(d.values(),k,a);
 J i=0; K x=ktn(KS,d.size());
 for(const auto& s:d.keys()) kS(x)[i++]=cs(s.c_str());
 return xD(x,y);
}

// ------------------------------------------------------------------------------------------------
// diagnostic functions -- check underlying pointers, storage data, reference counts, etc.
// ------------------------------------------------------------------------------------------------
// stordata - return storage data as k list (first move from cuda -> cpu, convert half to float)
// storinfo - return storage attributes & data as dictionary
// tensorinfo - return dictionary of attributes given tensor and detail flag
// ------------------------------------------------------------------------------------------------
static K stordata(const Tensor& a) {
 const auto& t=(a.dtype()==torch::kHalf) ? a.cpu().to(torch::kFloat) : a.cpu();
 const auto& d=t.dtype();
 const auto& s=t.storage();
 K x=ktn(maptype(d),s.nbytes() / d.itemsize());
 memcpy(kG(x),s.data(),s.nbytes());
 return x;
}

static K storinfo(const Tensor& t) {
 K x=KDICT,*k=&kK(x)[0],*v=&kK(x)[1];
 js(k, mapattr(Attr::bytes));    jk(v, kj(t.defined() ? t.storage().nbytes() : 0));
 js(k, mapattr(Attr::ref));      jk(v, kj(t.defined() ? storlong(t.storage(), Attr::ref) : 0));
 js(k, mapattr(Attr::ptr));      jk(v, kj(t.defined() ? storlong(t.storage(), Attr::ptr) : 0));
 js(k, mapattr(Attr::data));     jk(v, t.defined() ? stordata(t) : knk(0,0));
 return x;
}

K tensorinfo(const Tensor& t,bool d) {
 if(d && t.is_sparse()) {
  K x=KDICT,*a=&kK(x)[0],*b=&kK(x)[1];
  js(a, cs("indices")); jk(b, tensorinfo(t._indices(),d));
  js(a, cs("values"));  jk(b, tensorinfo(t._values(),d));
  return x;
 }
 K x=KDICT,*k=&kK(x)[0],*v=&kK(x)[1]; bool b=t.defined();
 js(k, mapattr(Attr::defined));     jk(v, kb(tensorflag(t, Attr::defined)));
 js(k, mapattr(Attr::device));      jk(v, ks(b ? tensorsym(t,Attr::device) : nullsym()));
 js(k, mapattr(Attr::dtype));       jk(v, ks(b ? tensorsym(t,Attr::dtype) : nullsym()));
 js(k, mapattr(Attr::layout));      jk(v, ks(b ? tensorsym(t,Attr::layout) : nullsym()));
 js(k, mapattr(Attr::gradient));    jk(v, ks(b ? tensorsym(t,Attr::gradient) : nullsym()));
 js(k, mapattr(Attr::pinned));      jk(v, ks(b ? tensorsym(t,Attr::pinned) : nullsym()));
 js(k, mapattr(Attr::memory));      jk(v, ks(b ? tensorsym(t,Attr::memory) : nullsym()));
 js(k, mapattr(Attr::leaf));        jk(v, kb(tensorflag(t, Attr::leaf)));
 js(k, mapattr(Attr::gradfn));      jk(v, ks(tensorsym(t,  Attr::gradfn)));
 js(k, mapattr(Attr::dim));         jk(v, kj(tensorlong(t, Attr::dim)));
 js(k, mapattr(Attr::sparsedim));   jk(v, kj(tensorlong(t, Attr::sparsedim)));
 js(k, mapattr(Attr::size));        jk(v, tensorsize(t,    Attr::size));
 js(k, mapattr(Attr::stride));      jk(v, tensorsize(t,    Attr::stride));
 js(k, mapattr(Attr::numel));       jk(v, kj(tensorlong(t, Attr::numel)));
 js(k, mapattr(Attr::itemsize));    jk(v, kj(tensorlong(t, Attr::itemsize)));
 js(k, mapattr(Attr::contiguous));  jk(v, kb(tensorflag(t, Attr::contiguous)));
 js(k, mapattr(Attr::coalesced));   jk(v, kb(tensorflag(t, Attr::coalesced)));
 js(k, mapattr(Attr::offset));      jk(v, kj(tensorlong(t, Attr::offset)));
 js(k, mapattr(Attr::ptr));         jk(v, kj(tensorlong(t, Attr::ptr)));
 js(k, mapattr(Attr::ref));         jk(v, kj(tensorlong(t, Attr::ref)));
 if(d) {
  js(k, mapattr(Attr::storage));   
  jk(v, storinfo(t));
 }
 return x;
}

// -------------------------------------------------------------------------------------------
// newsize - return new vector for tensor sizes, replacing size at dimension d with new value
// maxsize - find the maximum size at given dimension using underlying storage size
// checksize - check size of tensor/vector/dictionary input(s) along batching dimension
// -------------------------------------------------------------------------------------------
std::vector<int64_t> newsize(const Tensor& t,int64_t d,int64_t n) {
 auto v=t.sizes().vec(); v.at(d)=n; return v;
}

int64_t maxsize(const Tensor& t,int64_t d) {
 auto n=t.defined() ? t.storage().nbytes()/t.dtype().itemsize() : 0;
 auto e=n ? t.numel() : 0;
 return (n && e) ? t.size(d) * n / e : 0;
}

int64_t maxsize(const TensorVector& v,int64_t d) {
 int64_t i=0,m=-1;
 for(const auto&t:v) {
  auto n=maxsize(t,d);
  if(i) {
   TORCH_CHECK(m==n, "tensor[",i,"] size=",n,", but previous tensor(s) have size=",m," for dim ",d);
  } else {
   m=n;
  }
  ++i;
 }
 return m;
}

int64_t maxsize(const TensorDict& x,int64_t d) {
 int64_t m=-1;
 for(const auto& a:x.items()) {
  auto n=maxsize(a.value(),d);
  if(m<0)
   m=n;
  else
   TORCH_CHECK(m==n, "dictionary[",a.key(),"] size=",n,", but previous tensor(s) have size=",m," for dim ",d);
 }
 return m;
}

static int64_t checksize(const Input &x,int64_t n,int64_t d=0);
static int64_t checksize(const Input &x,int64_t n,int64_t d) {
 c10::visit(
  c10::overloaded(
   [&](const Tensor& x)       {auto m=maxsize(x,d); TORCH_CHECK(!n || m==n, "tensor size mismatch, ",n," vs ",m); n=m;},
   [&](const TensorVector& x) {for(const auto& t:x) n=checksize(t,n,d);},
   [&](const TensorDict& x)   {for(const auto& i:x.items()) n=checksize(i.value(),n,d);},
   [&](const Empty& x)        {}
  ),x);
 return n;
}

int64_t checksize(const Input& x,const Input& y) {
 return checksize(y,checksize(x,0));
}

// ----------------------------------------------------------------
// fullsize -  restore tensor(s) to maximum size at given dimension
// ----------------------------------------------------------------
int64_t fullsize(const Tensor& t,int64_t d,int64_t n) {
 if(n<0) n=maxsize(t,d);
 if(t.defined()) {
  if(t.size(d) != n)
   t.set_(t.storage(), 0, newsize(t,d,n), t.strides());
 }
 return n;
}

int64_t fullsize(const TensorVector& v,int64_t d,int64_t n) {
 if(n<0) n=maxsize(v,d);
 for(const auto& t:v) if(t.size(d) != n) fullsize(t,d,n);
 return n;
}

int64_t fullsize(const TensorDict& x,int64_t d,int64_t n) {
 if(n<0) n=maxsize(x,d);
 for(const auto& i:x.items()) if(i.value().size(d) != n) fullsize(i.value(),d,n);
 return n;
}

// -------------------------------------------------------------------------------------------
// batches - given batch size, total size & optional drop last flag, return number of batches
//   batch - use part of a tensor dimension, given batch index & size
//           requires tensor, batch index, optional dimension(default 0) & max length(default -1)
// -------------------------------------------------------------------------------------------
//int64_t batches(int64_t w,int64_t n,bool b) {return n%w ? n/w + !b : n/w;}
int64_t batches(int64_t w,int64_t n,bool b) {
 return n && w ? (n%w ? n/w + !b : n/w) : 0;
}

void batch(const Tensor& t,int64_t i,int64_t w,int64_t d=0,int64_t n=-1);
void batch(const Tensor& t,int64_t i,int64_t w,int64_t d,int64_t n) {
// const Tensor here for use w'dictionary items, e.g. for(auto& i:d.items()) batch(i.value(), ..)
 if(n<0) n=maxsize(t,d); // if not set, get max size of dimension d from overall storage size
 if(w>n) w=n;            // reduce subset window if greater than max size
 auto j=i*w;             // subset i -> offset j
 TORCH_CHECK(j < n,"batch[",i,"] invalid, valid range is from 0-",batches(w,n,true)-1);
 if(w>n-j) w=n-j;        // final subset may be a fraction of window
 t.set_(t.storage(), j*t.stride(d), w==t.size(d) ? t.sizes() : newsize(t,d,w), t.strides());
}

void batch(const Input& x,int64_t i,int64_t w,int64_t d,int64_t n) {
 if       (auto a=c10::get_if<Tensor>(&x))       { batch(*a,i,w,d,n);
 } else if(auto a=c10::get_if<TensorVector>(&x)) { for(auto &t:*a) batch(t,i,w,d,n);
 } else if(auto a=c10::get_if<TensorDict>(&x))   { for(auto &t:a->items()) batch(t.value(),i,w,d,n);
 } else if(       c10::get_if<Empty>(&x))        {
 } else { TORCH_ERROR("unrecognized input, unable to select batch");
 }
}

// ----------------------------------------------------------------------------
// nextbatch - return true if next batch available given tensor(s) & batch size
// ----------------------------------------------------------------------------
static bool nextbatch(const Tensor& t,int64_t w, int64_t d=0,int64_t n=-1);
static bool nextbatch(const Tensor& t,int64_t w, int64_t d,int64_t n) {
 TORCH_CHECK(t.dim(), "batch: tensor has no dimensions");
 if(n<0) n=maxsize(t,d);
 if(!n) return false;
 TORCH_CHECK(0<w && w<n, "batch: size ",w," not in the range 1-",n-1," (tensor size is ",n," for dimension ",d,")");
 if(n == t.size(d)) {
  batch(t,0,w,d,n);
  return true;
 } else {
  auto i=t.storage_offset() / (t.stride(d) * w) + 1;
  if(i < batches(w,n)) return batch(t,i,w,d,n), true;  // set next batch
  else                 return batch(t,0,n,d,n), false; // reset to full size
 }
}

static bool nextbatch(const TensorVector& x,int64_t w, int64_t d) {
 if(x.empty())
  return false;
 size_t r=0; auto n=maxsize(x,d);
 for(const auto& t:x)
  r+=nextbatch(t,w,d,n);
 TORCH_CHECK(!r || r==x.size(), "batch: only ",r," of ",x.size()," tensors successfully batched");
 return r ? true : false;
}

static bool nextbatch(const TensorDict& x,int64_t w, int64_t d) {
 if(x.is_empty())
  return false;
 size_t r=0; auto n=maxsize(x,d);
 for(const auto& a:x.items())
  r+=nextbatch(a.value(),w,d,n);
 TORCH_CHECK(!r || r==x.size(), "batch: only ",r," of ",x.size()," dictionary tensors successfully batched");
 return r ? true : false;
}

bool nextbatch(K x,int64_t w,int64_t d) {
 if(auto a=xten(x))               { return nextbatch(*a,w,d);
 } else if(auto a=xvec(x))        { return nextbatch(*a,w,d);
 } else if(auto a=xtensordict(x)) { return nextbatch(*a,w,d);
 } else { TORCH_ERROR("batch: expecting 1st arg of tensor,vector or dictionary, given ",kname(x));
 }
}

// ----------------------------------------------------------------------------
// batchindex - return true if next batch available given tensor(s) & batch size
// ----------------------------------------------------------------------------
void batchindex(K x,int64_t w,int64_t d,int64_t i) {
 if(auto a=xten(x))               { batch(*a,i,w,d);
 } else if(auto a=xvec(x))        { auto n=maxsize(*a,d); for(const auto& t:*a) batch(t,i,w,d,n);
 } else if(auto a=xtensordict(x)) { auto n=maxsize(*a,d); for(const auto& t:a->items()) batch(t.value(),i,w,d,n);
 } else { TORCH_ERROR("batch: expecting 1st arg of tensor,vector or dictionary, given ",kname(x));
 }
}

// -------------------------------------------------------------------------------------------
// setsafe - calls set_() after checking that the length implied by sizes & strides will fit
// subsetsafe - alternate form of subset using setsafe rather than maximum size dimension
// reset - [re]set tensor offset, size & storage
// -------------------------------------------------------------------------------------------
void setsafe(Tensor& t,int64_t i,const IntArrayRef& sz,const IntArrayRef& st) {
 const Storage& s=t.storage();
 TORCH_CHECK(s.nbytes()>=i+at::detail::computeStorageNbytes(sz,st,t.dtype().itemsize()), 
            "size ",sz," and stride ",st," require total of ",
             at::detail::computeStorageNbytes(sz,st,1),
            " plus offset of ",i," exceeds storage size of ",s.nbytes()/t.dtype().itemsize());
 t.set_(s,i,sz,st);
}

void subsetsafe(Tensor& t,int64_t d,int64_t i,int64_t w) {
 setsafe(t, i*t.stride(d), newsize(t,d,w), t.strides());
}

KAPI reset(K x) {
 KTRY
  Tensor t; J i;IntArrayRef y,z;
  TORCH_CHECK(xten(x,0,t), "reset: tensor expected as 1st of at least two arguments");
  TORCH_CHECK(xlong(x,1,i), "reset: offset (long) expected as 2nd argument, given ",kname(x,1));
  TORCH_CHECK(x->n<3 || xsize(x,2,y), "reset: size(s) expected as 3rd argument, given ",kname(x,2));
  TORCH_CHECK(x->n<4 || xsize(x,3,z), "reset: stride(s) expected as 4th argument, given ",kname(x,3));
  TORCH_CHECK(x->n>1 && x->n<5, "reset: up to 4 arguments expected, (tensor;offset;sizes;strides), but ",x->n," given");
  if(x->n==2)
   setsafe(t,i,t.sizes(),t.strides());            // existing size & stride, just move offset
  else if(x->n==3)
   setsafe(t,i,y,at::detail::defaultStrides(y));  // derive stride from size
  else
   setsafe(t,i,y,z);                              // specify offset, size & stride
  return (K)0;
 KCATCH("reset");
}

// --------------------------------------------------------------------------------
// narrow - narrow a tensor/vector along given dimension, according to offset,size
// --------------------------------------------------------------------------------
KAPI narrow(K x) {
 KTRY
  int64_t d,i,w;
  TORCH_CHECK(xint64(x,1,d) && xint64(x,2,i) && xint64(x,3,w) && x->n==4,
             "narrow: unrecognized arg(s), expecting (input; dim; offset; size)");
  if(auto *v=xvec(x,0)) {
   TensorVector r;
   for(const auto &t:*v) r.emplace_back(t.narrow(d,i,w));
    return kvec(r);
  } else if(auto *t=xten(x,0)) {
   return kten(t->narrow(d,i,w));
  } else {
   return kget(kput(x,0).narrow(d,i,w));
  }
 KCATCH("narrow");
}

// -------------------------------------------------------------------------------------------
// transpose tensor/array, 3-d & higher require two dim's to swap, optional in-place flag
// -------------------------------------------------------------------------------------------
KAPI transpose(K x) {
 KTRY
  if(x->t) return r1(x); // k scalar/simple list returned as is
  bool b=false; J n=x->n-xbool(x,x->n-1,b); int64_t i,j; Tensor t;
  bool p=xten(x,t) || xten(x,0,t);
  if(!p)
   t=xarray(x,4) ? (n=1,kput(x)) : kput(x,0);
  TORCH_CHECK(n==3 || t.dim()<3, "transpose of ",t.dim(),"d tensor needs the two dimenstions to be swapped");
  if(n==1)
   return p ? (b ? t.t_(),(K)0 : kten(t.t())) : kget(t.t());
  else if(xint64(x,1,i) && xint64(x,2,j) && n==3)
   return p ? (b ? t.transpose_(i,j),(K)0 : kten(t.transpose(i,j))) : kget(t.transpose(i,j));
  else
   TORCH_ERROR("transpose expects tensor or (tensor;inplace flag) or (tensor;dim1;dim2;optional inplace flag");
 KCATCH("transpose");
}

// -------------------------------------------------------------------------------------------
// inferflag - return true if a dimension needs to be derived (resize_ does not infer)
// infersize - if one dim is -1, infer size from other sizes and overall number of elements
// kresize1 - handle different methods for resizing, also allow inferred dim for resize_
// kresize2 - handle inputs for resize/View/reshape
// resize - resize tensor in place given size or tensor w'size to use, also allow k array input
//          will reallocate storage if larger size required, elements will be uninitialized
//          inferflag/infersize used to allow resize_() to infer a dimension
// view - attempt to create a new tensor that is view of existing tensor (shares storage)
//        error if view size is not compatible with input tensor's size and stride 
// reshape - like view, but will create new storage for new tensor if view not possible
// -------------------------------------------------------------------------------------------
static bool inferflag(const IntArrayRef& s) {for(auto i:s) if(i<0)return true; return false;}

static std::vector<int64_t> infersize(IntArrayRef s, int64_t m) {
  int64_t n=1; auto r=s.vec(); auto i=c10::optional<int64_t>();
  for (int64_t d=0, ndim = s.size(); d != ndim; d++) {
    if (s[d] == -1) {
      if (i)
        throw std::runtime_error("only one dimension can be inferred");
      i=d;
    } else if(s[d] >= 0) {
      n *= s[d];
    } else {
      TORCH_ERROR("invalid shape dimension ", s[d]);
    }
  }
  if (m == n || (i && n > 0 && m % n == 0)) {
    if (i) {
      TORCH_CHECK(n != 0, "cannot reshape tensor of 0 elements into shape ", s,
                  " because the unspecified dimension size -1 can be any value and is ambiguous");
      r[*i] = m / n;
    }
    return r;
  }
  TORCH_ERROR("shape ",s," is invalid for input of size ",m);
}

static K kresize1(I m,bool p,Tensor&& t, const IntArrayRef& s) {
 switch(m) {
  case 0:  t.resize_(inferflag(s) ? infersize(s,t.numel()) : s);
           return p ? (K)0 : kget(t);
  case 1:  return kresult(p,t.reshape(s));
  case 2:  return kresult(p,t.view(s));
  default: return KERR("invalid resize/reshape mode");
 }
}

static K kresize2(K x,I m,const char* e) {
 KTRY
  IntArrayRef n; Tensor *t=xten(x,0),*s=xten(x,1);
  TORCH_CHECK(!x->t,   e," not implemented for ",kname(x->t));
  TORCH_CHECK(x->n==2, e," expects (array/tensor;new size/tensor w'size to use)");
  TORCH_CHECK(s || xsize(x,1,n), e," expects 2nd arg of size or tensor w'size to use");
  return kresize1(m, t, t ? *t : kput(x,0), s ? s->sizes() : n);
 KCATCH(e);
}

KAPI resize(K x)  {return kresize2(x, 0, "resize");}
KAPI reshape(K x) {return kresize2(x, 1, "reshape");}
KAPI view(K x)    {return kresize2(x, 2, "view");}

// ----------------------------------------------------------------------------
// tensorpick - pick tensor(s) from vector, dictionary or module, return vector
// ----------------------------------------------------------------------------
static TensorVector tensorpick(K x,bool u,const char* e,const Tensor& t) { 
 IntArrayRef n; TensorVector z;
 TORCH_CHECK(xsize(x,n), e,": expecting 2nd arg of tensor indices, given ",kname(x));
 TORCH_CHECK(u || t.defined(), e,": tensor is not defined");
 TORCH_CHECK(t.dim(), e,": 0-dim tensor");
 for(auto i:n) {
  TORCH_CHECK(-1<i && i<t.size(0),  e,": tensor[",i,"] invalid given a ",t.size(0),"-element tensor");
  z.push_back(t[i]);
 }
 return z;
}

static TensorVector tensorpick(K x,bool u,const char* e,const TensorVector& v) { 
 IntArrayRef n; TensorVector z; int64_t m=v.size();
 TORCH_CHECK(xsize(x,n), e,": expecting 2nd arg of vector indices, given ",kname(x));
 for(auto i:n) {
  TORCH_CHECK(-1<i && i<m,         e,": vector[",i,"] invalid given a ",m,"-element vector");
  TORCH_CHECK(u || v[i].defined(), e,": vector[",i,"] is not defined");
  z.push_back(v[i]);
 }
 return z;
}

static TensorVector tensorpick(K x,bool u,const char *e,const TensorDict& d) {
 SymArrayRef a; TensorVector z;
 TORCH_CHECK(xsyms(x,a), e,": unable to get dictionary key(s) given ",kname(x));
 for(auto s:a) {
  auto *t=d.find(s);
  TORCH_CHECK(t, e,": dictionary key `",s," not found");
  TORCH_CHECK(u || t->defined(), e,": dictionary tensor for key `",s," not defined");
  z.push_back(*t);
 }
 return z;
}

static TensorVector tensorpick(K x,bool u,Cast c,const char *e,const Module& m) {
 SymArrayRef a; TensorVector z;
 TORCH_CHECK(xsyms(x,a), e,": unable to get ",tensortype(c)," names(s) given ",kname(x));
 for(auto s:a) {
  auto *t=findtensor(m,s,c);
  TORCH_CHECK(t,                 e,": ",(c==Cast::tensor ? "parameter/buffer" : tensortype(c))," `",s," not found");
  TORCH_CHECK(u || t->defined(), e,": ",(c==Cast::tensor ? "parameter/buffer" : tensortype(c))," `",s," not defined");
  z.push_back(*t);
 }
 return z;
}

TensorVector tensorpick(Ktag *g,K x,bool u,Cast c,const char* e) {
 // u-flag set true if undefined tensors allowed, c-cast(parameter,buffer or tensor)
 switch(g->a) {
  case Class::tensor:     return tensorpick(x,u,e,g->tensor());
  case Class::vector:     return tensorpick(x,u,e,g->vector());
  case Class::dict:       return tensorpick(x,u,e,g->dict());
  case Class::loss:
  case Class::module:
  case Class::optimizer:
  case Class::model:      return tensorpick(x,u,c,e,g->module());
  default: TORCH_ERROR(e,": not implemented for ",mapclass(g->a));
 }
}

// ----------------------------------------------------------------------------
// fntype - return non-linearity type given symbol
// gain - return recommended gain given activation fn & optional parameter
// ----------------------------------------------------------------------------
using Fan=nn::init::FanModeType;
using Func=nn::init::NonlinearityType;

static bool fantype(S s,Fan& f) {
 bool b=true;
 switch(emap(s)) {
  case Enum::fanin:  f=torch::kFanIn; break;
  case Enum::fanout: f=torch::kFanOut; break;
  default: b=false; break;
 }
 return b;
}

static bool fntype(S s,Func& f) {
 bool b=true;
 switch(emap(s)) {
  case Enum::linear:          f=torch::kLinear; break;
  case Enum::conv1d:          f=torch::kConv1D; break;
  case Enum::conv2d:          f=torch::kConv2D; break;
  case Enum::conv3d:          f=torch::kConv3D; break;
  case Enum::convtranspose1d: f=torch::kConvTranspose1D; break;
  case Enum::convtranspose2d: f=torch::kConvTranspose2D; break;
  case Enum::convtranspose3d: f=torch::kConvTranspose3D; break;
  case Enum::sigmoid:         f=torch::kSigmoid; break;
  case Enum::tanh:            f=torch::kTanh; break;
  case Enum::relu:            f=torch::kReLU; break;
  case Enum::leakyrelu:       f=torch::kLeakyReLU; break;
  default: b=false; break;
 }
 return b;
}

KAPI gain(K x) {
 KTRY
  S s; double d=0.01; Func f;
  TORCH_CHECK(xsym(x,s) || (xsym(x,0,s) && x->n==2 && xnum(x,1,d)),
              "gain: expecting function name and optional parameter, e.g. `relu or (`leakyrelu;0.2)");
  TORCH_CHECK(fntype(s,f), "gain: unrecognized non-linearity `",s);
  return kf(nn::init::calculate_gain(f,d));
 KCATCH("gain");
}

// ----------------------------------------------------------------------------
// kfill - handle inputs to fill tensor(s) in place with 0's or 1's or identity
// zeros - zero-fill tensor(s) given array,tensor,vector,dict or module
// ones - fill tensor(s) w'ones given array,tensor,vector,dict or module
// eye - identity fill given tensor(s)
// dirac - fills 3,4,5-dim tensor with Dirac delta function (groups=1 only)
// orthogonal - fills tensor with semi-orthogonal matrix w'optional gain
// xnormal,xuniform - xavier normal & uniform initialization w'optional gain
// ----------------------------------------------------------------------------
enum class Fill:char {zero=0,one,eye,dirac,orthogonal,xnormal,xuniform};

static void kfill(const char *c,Fill f,Tensor& t) {
 TORCH_CHECK(t.defined(), c,": not implemented for undefined tensor");
 switch(f) {
  case Fill::zero:       t.zero_(); break;
  case Fill::one:        t.fill_(1); break;
  case Fill::eye:        nn::init::eye_(t); break;
  case Fill::dirac:      nn::init::dirac_(t); break;
  case Fill::orthogonal: nn::init::orthogonal_(t); break;
  case Fill::xnormal:    nn::init::xavier_normal_(t); break;
  case Fill::xuniform:   nn::init::xavier_uniform_(t); break;
 }
}

static void kfill(const char *c,double d,Fill f,Tensor &t) {
 TORCH_CHECK(t.defined(), c,": not implemented for undefined tensor");
 switch(f) {
  case Fill::orthogonal: nn::init::orthogonal_(t,d); break;
  case Fill::xnormal:    nn::init::xavier_normal_(t,d); break;
  case Fill::xuniform:   nn::init::xavier_uniform_(t,d); break;
  default: TORCH_ERROR(c,": not implemented for additional gain argument of ",d);
 }
}

static K kfill(K x,bool a,const char* c,Fill f) {
// a-true if allows optional double arg at end: gain
 KTRY
  torch::NoGradGuard nograd; double d;
  if(auto *g=xtag(x)) {
   switch(g->a) {
    case Class::tensor: kfill(c,f,g->tensor()); break;
    case Class::vector: for(auto& t:g->vector()) kfill(c,f,t); break;
    case Class::dict:   for(auto& t:g->dict().values()) kfill(c,f,t); break;
    default:TORCH_ERROR(c,": not implemented for single ",mapclass(g->a)," argument"); break;
   }
   return (K)0;
  } else if(auto *g=xtag(x,0)) {
   if(x->n==2) {
    if(xdouble(x,1,d)) {
     switch(g->a) {
      case Class::tensor: kfill(c,d,f,g->tensor()); break;
      case Class::vector: for(auto& t:g->vector()) kfill(c,d,f,t); break;
      case Class::dict:   for(auto& t:g->dict().values()) kfill(c,d,f,t); break;
      default:TORCH_ERROR(c,": not implemented for single ",mapclass(g->a)," argument"); break;
     }
    } else {
     for(auto& t:tensorpick(g,kK(x)[1],false,Cast::tensor,c))
      kfill(c,f,t);
    }
   } else if(x->n==3) {
    TORCH_CHECK(a, c,": no more than 2 args expected, 3 supplied");
    TORCH_CHECK(xdouble(x,2,d), c,": unable to read final arg of gain, double expected, given ",kname(x,2));
    for(auto& t:tensorpick(g,kK(x)[1],false,Cast::tensor,c))
     kfill(c,d,f,t);
   } else {
    TORCH_ERROR(c,": given ",mapclass(g->a),", expecting up to ",2+a," args, but ",x->n," args supplied");
   }
   return (K)0;
  } else {
   Tensor t;
   if(!x->t && x->n==2 && xdouble(x,1,d)) {
    t=kput(x,0); kfill(c,d,f,t);
   } else {
    t=kput(x); kfill(c,f,t);
   }
   return kget(t);
  }
 KCATCH(c);
}

KAPI zeros(K x) {return kfill(x, false, "zeros", Fill::zero);}
KAPI  ones(K x) {return kfill(x, false, "ones",  Fill::one);}
KAPI   eye(K x) {return kfill(x, false, "eye",   Fill::eye);}
KAPI dirac(K x) {return kfill(x, false, "dirac", Fill::dirac);}

// allows optional argument at end input, (input;arg), (input;indices/names) or (input;indices/keys;arg)
KAPI orthogonal(K x) {return kfill(x, true, "orthogonal",     Fill::orthogonal);}
KAPI    xnormal(K x) {return kfill(x, true, "xaiver normal",  Fill::xnormal);}
KAPI   xuniform(K x) {return kfill(x, true, "xaiver uniform", Fill::xuniform);}

// ---------------------------------------------------------------------------
// kaimarg - process k arg(s) for fan in/out, nonlinearity & parameter
// kaiming - process k arg(s) for kaiming normal & unform initialization
// knormal,kuniform - api fns for kaiming normal & uniform initialization
// ---------------------------------------------------------------------------
using Kaiming=Tensor (*)(Tensor,double,Fan,Func);

static void kaimarg(K x,J i,const char *c,double& d,Fan& f,Func& n) {
 for(; i<x->n; ++i) {
  K y=kK(x)[i];
  if(y->t == -KS) {
   TORCH_CHECK(fantype(y->s,f) || fntype(y->s,n), c,": unrecognized symbol `",y->s);
  } else {
   TORCH_CHECK(xnum(y,d), c,": arg[",i,"] unrecognized, ",kname(y));
  }
 }
}

static void kaiming(Ktag *g,const char *c,double d,const Fan& f,const Func& n,Kaiming k) {
 switch(g->a) {
  case Class::tensor: k(g->tensor(),d,f,n); break;
  case Class::vector: for(auto& t:g->vector()) k(t,d,f,n); break;
  case Class::dict:   for(auto& t:g->dict().values()) k(t,d,f,n); break;
  default:TORCH_ERROR(c,": not implemented for single ",mapclass(g->a)," argument"); break;
 }
}

static K kaiming(K x,const char* c,Kaiming k) {
 KTRY
  torch::NoGradGuard nograd; 
  TORCH_CHECK(!x->t, c,": not implemented for ",kname(x));
  double d=0; Fan f=torch::kFanIn; Func n=torch::kLeakyReLU;
  if(auto *g=xtag(x)) {
   kaiming(g,c,d,f,n,k);
   return (K)0;
  } else if(auto *g=xtag(x,0)) {
   TORCH_CHECK(1<x->n && x->n<6, c,": 2-5 args expected but ",x->n," supplied");
   J i; K y=kK(x)[1];
   switch(g->a) {
    case Class::tensor:
    case Class::vector:
     i=(y->t == -KJ || y->t == KJ) ? 2 : 1; // offset for other args
     kaimarg(x,i,c,d,f,n);                  // process other args
     if(i==2) {
      for(auto& t:tensorpick(g,y,false,Cast::tensor,c))
       k(t,d,f,n);
     } else {
       kaiming(g,c,d,f,n,k);
     }
     break;
    default:
     i=(y->t == -KS || y->t == KS) ? 2 : 1;
     TORCH_CHECK(i==2 || g->a==Class::dict, c,": tensor name(s) expected with supplied ",mapclass(g->a));
     if(g->a==Class::dict && i==2 && y->t==-KS && (fantype(y->s,f) || fntype(y->s,n))) i=1;
     kaimarg(x,i,c,d,f,n);
     if(i==2) {
      for(auto& t:tensorpick(g,y,false,Cast::tensor,c))
       k(t,d,f,n);
     } else {
       kaiming(g,c,d,f,n,k);
     }
     break;
   }
   return (K)0;
  } else {
   Tensor t;
   if(xarray(x,2)) {
    t=kput(x);
   } else {
    t=kput(x,0); kaimarg(x,1,c,d,f,n);
   }
   k(t,d,f,n);
   return kget(t);
  }
 KCATCH(c);
}

KAPI  knormal(K x) {return kaiming(x, "kaiming normal",  nn::init::kaiming_normal_);}
KAPI kuniform(K x) {return kaiming(x, "kaiming uniform", nn::init::kaiming_uniform_);}

// ---------------------------------------------
// snormal - sparse initialization
// ---------------------------------------------
KAPI snormal(K x) {
 KTRY
  torch::NoGradGuard nograd;
  double f,d=.01; const char *c="sparse initialize";
  TORCH_CHECK(!x->t,  c,": not implemented for ",kname(x));
  TORCH_CHECK(!xarray(x,3), c,": unrecognized arg(s), expecting (input;sparse fraction), (input;indices/names;fraction), etc.");
  TORCH_CHECK(1<x->n && x->n<5, c,": expecting 2-4 args, ",x->n," supplied");
  if(auto *g=xtag(x,0)) {
   if(xdouble(x,1,f) && (x->n==2 || (x->n==3 && xdouble(x,2,d)))) {
    switch(g->a) {
     case Class::tensor: nn::init::sparse_(g->tensor(),f,d); break;
     case Class::vector: for(auto& t:g->vector()) nn::init::sparse_(t,f,d); break;
     case Class::dict:   for(auto& t:g->dict().values()) nn::init::sparse_(t,f,d); break;
     default:TORCH_ERROR(c,": not implemented for ",mapclass(g->a)," argument without names"); break;
    }
   } else {
    TORCH_CHECK(x->n>=3, c,": expecting 3 args, ",mapclass(g->a),", indices/names and sparse fraction, but ",x->n," args supplied");
    TORCH_CHECK(xdouble(x,2,f), c,": expecting 3rd arg of sparse fraction as double, given ",kname(x,2));
    TORCH_CHECK(x->n==3 || xdouble(x,3,d), c,": expecting 4th arg of std deviation as double, given ",kname(x,3));
    for(auto& t:tensorpick(g,kK(x)[1],false,Cast::tensor,c))
     nn::init::sparse_(t,f,d);
   }
   return (K)0;
  } else {
   TORCH_CHECK(x->n==2 || x->n==3, "sparse initialize: expecting array with sparse fraction and optional stddev, but ",x->n," args supplied");
   TORCH_CHECK(xdouble(x,1,f), "sparse initialize: expecting 2nd arg of sparse fraction to be double, given ",kname(x,1));
   TORCH_CHECK(x->n==2 || xdouble(x,2,d), "sparse initialize: expecting 3rd arg of stddev to be double, given ",kname(x,2));
   return kget(nn::init::sparse_(kput(x,0),f,d));
  }
 KCATCH("sparse initialize");
}

// ---------------------------------------------
// fill - fill  tensor/array with given element
// ---------------------------------------------
KAPI fill(K x) {
 KTRY
  torch::NoGradGuard nograd;
  TORCH_CHECK(!x->t, "fill: not implemented for ",kname(x));
  TORCH_CHECK(1<x->n && x->n<4, "fill: expecting 2-3 args, ",x->n," supplied");
  K y=kK(x)[x->n-1];
  TORCH_CHECK(y->t<0, "fill: unable to get fill element from final arg of ",kname(y));
  const auto& n=kput(y).item();
  if(auto *g=xtag(x,0)) {
   K y=kK(x)[1];
   switch(g->a) {
    case Class::tensor:
     if(x->n==2) {
      g->tensor().fill_(n);
     } else {
      for(auto& t:tensorpick(y,false,"fill",g->tensor()))
       t.fill_(n);
     }
     break;
    case Class::vector:
     for(auto& t:x->n==2 ? g->vector() : tensorpick(y,false,"fill",g->vector()))
      t.fill_(n);
     break;
    case Class::dict:
     for(auto& t:x->n==2 ? g->dict().values() : tensorpick(y,false,"fill",g->dict()))
      t.fill_(n);
     break;
    case Class::loss:
    case Class::module:
    case Class::optimizer:
    case Class::model:
     TORCH_CHECK(x->n==3, "fill: given ",mapclass(g->a)," expecting parameter/buffer names in addition to fill element");
     for(auto& t:tensorpick(y,false,Cast::tensor,"fill",g->module()))
      t.fill_(n);  
     break;
    default:
     TORCH_ERROR("fill: not implemented for ",mapclass(g->a)); 
     break;
   }
   return (K)0;
  } else {
   if(x->n==2) {
    return kget(kput(x,0).fill_(n));  //read in array->tensor, fill, return as array
   } else {
    auto t=kput(x,0); K y=kK(x)[1];   //read in array -> tensor, use indices to fill, return array
    for(auto& a:tensorpick(y,false,"fill",t))
     a.fill_(n);
    return kget(t);
   }
  }
 KCATCH("fill");
}

// -------------------------------------------------------------
// filldiagonal - fill diagonal of matrix input with given value
// -------------------------------------------------------------
KAPI filldiagonal(K x) {
 KTRY
  torch::NoGradGuard nograd;
  bool w=false; Scalar s;
  if(xnum(x,1,s) && (x->n==2 || (x->n==3 && xbool(x,2,w))))
   if(auto *t=xten(x,0))
    return t->fill_diagonal_(s,w), (K)0;
   else
    return kget(kput(x,0).fill_diagonal_(s,w));
  else
   TORCH_ERROR("fill diagonal expects (tensor/array;fill element;optional wrap flag)");
 KCATCH("fill diagonal");
}

// ------------------------------------------------------------------------------------------
// imagegrid - rearrange input images into a single tensor w'images across given cols & rows
// makegrid - api fn to accept array/tensor and rows,cols,padding,pad value -> grid of images
// ------------------------------------------------------------------------------------------
Tensor imagegrid(const Tensor& a,int64_t r,int64_t c,int64_t p,short v) {
 Tensor t=a;
 if(t.dim()==2) t=t.unsqueeze(0);           // single image H x W
 if(t.dim()==3) {                           // single image C x H x W
  if(t.size(0)==1) t=torch::cat({t,t,t},0); // if single-channel, convert to 3-channel
  t=t.unsqueeze(0);
 }
 if(t.dim()==4 && t.size(1)==1)
  t=torch::cat({t,t,t},1);                  // n single-channel images
 int64_t n=t.size(0),h=t.size(2)+p,w=t.size(3)+p,i,j,k=0;
 auto g=t.new_full({t.size(1),r*h+p,c*w+p},v);
 for(i=0; i<r; ++i) {
  for(j=0; j<c; ++j) {
   if(k>=n) break;
   g.narrow(1,i*h+p,h-p).narrow(2,j*w+p,w-p).copy_(t[k++]);
  }
 }
 return g;
}

KAPI makegrid(K x,K y,K z) {
 KTRY
  bool b; int64_t r=nj,c=nj,p=2,v=0; Tensor t;
  if(xint64(x,1,r) && (x->n==2 || (xint64(x,2,c) && (x->n==3 || (xint64(x,3,p) && (x->n==4 || (xint64(x,4,v) && x->n==5))))))) {
   if(!(b=xten(x,0,t))) t=kput(x,0);
   TORCH_CHECK(t.dim()==3 || t.dim()==4, "makegrid: expecting 3 or 4-dimensional input, ",t.dim()," dimension(s) given");
   TORCH_CHECK(r !=nj || c !=nj, "makegrid: both rows & columns cannot be null");
   if(t.dim()==3) t=t.unsqueeze(1);
   return kresult(b,imagegrid(t,r,c,p,v));
  } else {
   TORCH_ERROR("makegrid: unrecognized arg(s), expecting 2-5 args, (input array/tensor; rows; cols; padding; pad value)");
  }
 KCATCH("makegrid");
}

// ---------------------------------------------------------------------------------------
// tensorcopy - tgt <- src values, must have same type & device, tgt resized if src larger
// amend - tensor values are amended, respecting current device & type
// ---------------------------------------------------------------------------------------
void tensorcopy(Tensor &t,const Tensor &s,bool a) {
 if(s.dtype() != t.dtype()) {
  TORCH_ERROR("unable to copy values from ",s.dtype()," tensor to ",t.dtype()," tensor");
 } else if(s.device() != t.device()) {
  TORCH_ERROR("unable to copy values across devices, from ",s.device()," to ",t.device());
 } else {
  t.resize_as_(s).copy_(s,a);
 }
}

KAPI amend(K x) {
 KTRY
  torch::NoGradGuard ng;
  bool a=false; auto *g=xtag(x,0); S s; int64_t i; Tensor *t;
  TORCH_CHECK(g, "amend: expecting 1st arg of tensor,vector or dictionary");
  switch(g->a) {
   case Class::tensor:
    TORCH_CHECK(x->n==2 || (x->n==3 && xbool(x,2,a)),
               "amend: 2-3 arguments expected, e.g. (tensor;values) or (tensor;values;asyncflag), but ",x->n," args given");
    t=xten(x,1); g->tensor().copy_(t ? *t : kput(kK(x)[1]), a);
    break; 
   case Class::vector: {
    auto& v=g->vector();
    TORCH_CHECK(x->n==3 || (x->n==4 && xbool(x,3,a)),
               "amend: 3-4 arguments expected, e.g. (vector;index;values) or (vector;index;values;asyncflag), but ",x->n," given");
    TORCH_CHECK(xint64(x,1,i), "amend: given vector, 2nd argument is an index(long), given ",kname(x,1));
    TORCH_CHECK(-1<i && i<(int64_t)v.size(), "amend: vector[",i,"] is invalid given a ",v.size(),"-element vector");
    t=xten(x,1); v[i].copy_(t ?*t : kput(kK(x)[2]), a);
    break;   
   }
   case Class::dict:
    TORCH_CHECK(x->n==3 || (x->n==4 && xbool(x,3,a)),
               "amend: 3-4 arguments expected, e.g. (dictionary;key;values) or (dictionary;key;values;asyncflag), but ",x->n," given");
    TORCH_CHECK(xsym(x,1,s), "amend: given dictionary, 2nd argument is a key(symbol), given ",kname(x,1));
    t=xten(x,1); g->dict()[s].copy_(t ?*t : kput(kK(x)[2]), a);
    break;   
   default:
    TORCH_ERROR("amend: not implemented for ",mapclass(g->a));
  }
  return (K)0;
 KCATCH("amend");
}

// ------------------------------------------------------------------------------------------
// kgrad - return gradient data or empty, if ptr enlisted, return gradient ptr (must free)
// gradflag - return requires gradient boolean (as opposed to sym of `grad/`nograd)
// ------------------------------------------------------------------------------------------
KAPI kgrad(K x) {
 KTRY
  bool p=false; Tensor t;
  if(xten(x,t) || (p=(xten(x,0,t) && x->n==1))) {
   if(p) return t.grad().defined() ? kten(t.grad()) : KERR("no gradient defined");
   else  return t.grad().defined() ? kget(t.grad()) : (K)0;
 } else {
  return KERR("unexpected arg(s) for gradient, expecting tensor (enlist to return gradient ptr)");
 }
 KCATCH("unable to get gradient");
}

KAPI gradflag(K x) {
 KTRY
  bool b; Ktag *k=xtag(x); if(!k) k=xtag(x,0); SymArrayRef s; IntArrayRef i;
  if(k && x->n==1) {
   return attr(x, -KB, Attr::gradflag);
  } else if(k && x->n==2 && xbool(x,1,b)) {
   switch(k->a) {
    case Class::tensor: k->tensor().set_requires_grad(b); break;
    case Class::vector: for(auto& t:k->vector()) t.set_requires_grad(b); break;
    case Class::dict: for(auto& t:k->dict().values()) t.set_requires_grad(b); break;
    case Class::module:
    case Class::model:  for(auto& t:k->module().parameters()) t.set_requires_grad(b); break;
    default: TORCH_ERROR("gradflag: not implemented for ",mapclass(k->a));
   }
   return (K)0;
  } else if(k && xsyms(x,1,s) && (x->n==2 || (x->n==3 && xbool(x,2,b)))) {
   TORCH_CHECK(k->a==Class::dict, "gradflag: 2nd arg of key(s) not valid with ",mapclass(k->a));
   const auto& d=k->dict();
   for(const auto& a:s) TORCH_CHECK(d.contains(a), "gradflag: dict[`",a,"] not found");
   if(x->n==2) {
    if(kK(x)[1]->t == -KS) // scalar key returns scalar  
     return kb(d[s[0]].requires_grad());
    J i=0; K y=ktn(KS,s.size()), z=ktn(KB,s.size());
    for(const auto& a:s) kS(y)[i]=a, kG(z)[i++]=d[a].requires_grad();
    return xD(y,z);
   } else {
    for(const auto& k:s) d[k].set_requires_grad(b);
    return (K)0;
   }
  } else if(k && xsize(x,1,i) && (x->n==2 || (x->n==3 && xbool(x,2,b)))) {
   TORCH_CHECK(k->a==Class::vector, "gradflag: 2nd arg of index/indices not valid with ",mapclass(k->a));
   const auto& v=k->vector(); int64_t n=v.size();
   for(const auto& j:i) TORCH_CHECK(-1<j && j<n, "gradflag: vector[",j,"] invalid for ",n,"-element vector");
   if(x->n==2) {
    if(kK(x)[1]->t == -KJ) // scalar index returns scalar flag
     return kb(v[i[0]].requires_grad());
    K r=ktn(KB,i.size()); J ij=0;
    for(const auto& j:i) kG(r)[ij++]=v[j].requires_grad();
    return r;
   } else {
    for(const auto& j:i) v[j].set_requires_grad(b);
    return (K)0;
   }
  } else {
   TORCH_ERROR("gradflag: unrecognized arg(s), expecting tensor/vector/dictionary or (object;flag) or (dict/vec;keys/inds) or (dict/vec;keys/inds;flag)");
  }
 KCATCH("gradflag");
}

// --------------------------------------------------------------------------------------
// detach - k api fn to detach tensor/vector/dictonary  
//          tensor has optional inplace flag, vec/dict allows additional indices/keys
// --------------------------------------------------------------------------------------
KAPI detach(K x) {
 KTRY
  Ktag *g=xtag(x); bool a=g,b=false;
  TORCH_CHECK(g||(g=xtag(x,0)), "detach: expecting tensor, vector or dictionary");
  switch(g->a) {
   case Class::tensor: {
    TORCH_CHECK(a || (x->n==2 && xbool(x,1,b)), "detach: expecting tensor or (tensor; inplace flag)");
    return b ? g->tensor().detach_(),(K)0 : kten(g->tensor().detach());
   }
   case Class::vector: {
    if(a) {
     for(auto &t:g->vector()) t.detach_();
    } else {
     IntArrayRef i; auto& v=g->vector(); int64_t n=v.size();
     TORCH_CHECK(x->n==2, "detach: vector supplied, but unexpected number of additional args(",x->n-1,")");
     TORCH_CHECK(xsize(x,1,i), "detach: vector supplied, expected 2nd arg of indices, given ",kname(x,1));
     for(auto j:i) {
      TORCH_CHECK(j>-1 && j<n, "detach: vector[",j,"] out of bounds");
      v[j].detach_();
     }
    }
    return (K)0;
   }
   case Class::dict: {
    if(a) {
     for(auto &a:g->dict()) a.value().detach_();
    } else {
     SymArrayRef s; auto& d=g->dict();
     TORCH_CHECK(x->n==2, "detach: dictionary supplied, but unexpected number of additional args(",x->n-1,")");
     TORCH_CHECK(xsyms(x,1,s), "detach: dictionary supplied, expected 2nd arg of key(s), given ",kname(x,1));
     for(auto k:s) {
      TORCH_CHECK(d.contains(k), "detach: dictionary[`",k,"] not found");
      d[k].detach_();
     }
    }
    return (K)0;
   }
   default: TORCH_ERROR("detach: not implemented for ",mapclass(g->a));
  }
  return (K)0;
 KCATCH("detach");
}

// -------------------------------------------------
// ksame - check if same tensor/storage
// same - check if underlying tensor pointers match
// alias - check if same underlying storage
// -------------------------------------------------
static K ksame(K x,bool s) {
 KTRY
  Tensor *a=xten(x,0),*b=xten(x,1);
  TORCH_CHECK(a && b && x->n==2, (s ? "same" : "alias"), ": expects two tensors");
  return kb(s ? a->is_same(*b) : a->is_alias_of(*b));
 KCATCH("same/alias");
}

KAPI same(K x)  {return ksame(x,true);}
KAPI alias(K x) {return ksame(x,false);}

// ----------------------------------
// tensor fns defined in k namespace
// ----------------------------------
void tensorfn(K x) {
 fn(x, "alias",        KFN(alias),         1);
 fn(x, "amend",        KFN(amend),         1);
 fn(x, "cat",          KFN(cat),           1);
 fn(x, "coalesce",     KFN(coalesce),      1);
 fn(x, "dense",        KFN(dense),         1);
 fn(x, "detach",       KFN(detach),        1);
 fn(x, "dict",         KFN(dict),          1);
 fn(x, "dirac",        KFN(dirac),         1);
 fn(x, "expand",       KFN(expand),        1);
 fn(x, "eye",          KFN(eye),           1);
 fn(x, "filldiagonal", KFN(filldiagonal),  1);
 fn(x, "fill",         KFN(fill),          1);
 fn(x, "gain",         KFN(gain),          1);
 fn(x, "gradflag",     KFN(gradflag),      1);
 fn(x, "grad",         KFN(kgrad),         1);
 fn(x, "imag",         KFN(imag),          1);
 fn(x, "indices",      KFN(indices),       1);
 fn(x, "index",        KFN(kindex),        1);
 fn(x, "isreal",       KFN(isreal),        1);
 fn(x, "knormal",      KFN(knormal),       1);
 fn(x, "kuniform",     KFN(kuniform),      1);
 fn(x, "makegrid",     KFN(makegrid),      1);
 fn(x, "narrow",       KFN(narrow),        1);
 fn(x, "ones",         KFN(ones),          1);
 fn(x, "orthogonal",   KFN(orthogonal),    1);
 fn(x, "permute",      KFN(permute),       1);
 fn(x, "real",         KFN(real),          1);
 fn(x, "reset",        KFN(reset),         1);
 fn(x, "reshape",      KFN(reshape),       1);
 fn(x, "resize",       KFN(resize),        1);
 fn(x, "same",         KFN(same),          1);
 fn(x, "snormal",      KFN(snormal),       1);
 fn(x, "sparseindex",  KFN(sparseindex),   1);
 fn(x, "sparse",       KFN(sparse),        1);
 fn(x, "squeeze",      KFN(squeeze),       1);
 fn(x, "stack",        KFN(stack),         1);
 fn(x, "tensor",       KFN(tensor),        1);
 fn(x, "transpose",    KFN(transpose),     1);
 fn(x, "unsqueeze",    KFN(unsqueeze),     1);
 fn(x, "values",       KFN(values),        1);
 fn(x, "vector",       KFN(vector),        1);
 fn(x, "View",         KFN(view),          1);
 fn(x, "xnormal",      KFN(xnormal),       1);
 fn(x, "xuniform",     KFN(xuniform),      1);
 fn(x, "zeros",        KFN(zeros),         1);
}

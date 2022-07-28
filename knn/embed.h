#pragma once
#include "util.h"

namespace knn {

// -----------------------------------------------------------------------------
// EmbedPositionOptions - options for position embedding
// EmbedPosition - learn position embedding: rows(max sequence), cols(embed dim)
// -----------------------------------------------------------------------------
struct TORCH_API EmbedPositionOptions {
 EmbedPositionOptions(int64_t r,int64_t c) : rows_(r),cols_(c) {}
 TORCH_ARG(int64_t, rows);
 TORCH_ARG(int64_t, cols);
};

class TORCH_API EmbedPositionImpl : public torch::nn::Cloneable<EmbedPositionImpl> {
 public:
 EmbedPositionImpl(int64_t r,int64_t c) : EmbedPositionImpl(EmbedPositionOptions(r,c)) {}
 explicit EmbedPositionImpl(const EmbedPositionOptions&);
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& x);
 EmbedPositionOptions options;
 Tensor pos;
};
TORCH_MODULE(EmbedPosition);

// -----------------------------------------------------
// embedpos - set/get options for position embedding
// -----------------------------------------------------
EmbedPositionOptions embedpos(K,J,Cast);
K embedpos(bool,const EmbedPositionOptions&);

// -----------------------------------------------------------------------------
// EmbedSequenceOptions - options for embedding tokens & position in sequence
// EmbedSequence - learn token & position embedding given rows,cols,length
//                 rows - vocabulary size for tokens
//                 cols - embedding dimension (attributes for each token)
//                 length - maximum sequence length
// -----------------------------------------------------------------------------
struct TORCH_API EmbedSequenceOptions {
 EmbedSequenceOptions(int64_t r,int64_t c,int64_t n) : rows_(r),cols_(c),length_(n) {}
 TORCH_ARG(int64_t, rows);
 TORCH_ARG(int64_t, cols);
 TORCH_ARG(int64_t, length);
};

class TORCH_API EmbedSequenceImpl : public torch::nn::Cloneable<EmbedSequenceImpl> {
 public:
 EmbedSequenceImpl(int64_t r,int64_t c,int64_t n) : EmbedSequenceImpl(EmbedSequenceOptions(r,c,n)) {}
 explicit EmbedSequenceImpl(const EmbedSequenceOptions&);
 void reset() override;
 void pretty_print(std::ostream& s) const override;
 Tensor forward(const Tensor& x);
 EmbedSequenceOptions options;
 torch::nn::Embedding tok = nullptr;
 knn::EmbedPosition   pos = nullptr;
};
TORCH_MODULE(EmbedSequence);

// ---------------------------------------------------------
// embedseq - set/get options for token & position embedding
// ---------------------------------------------------------
EmbedSequenceOptions embedseq(K,J,Cast);
K embedseq(bool,const EmbedSequenceOptions&);

// --------------------------------------------------------------------------------------
// embedset - set name/value pairs specific to Embedding vs EmbeddingBag
// embedpair - handle name/value pairs for both types of embedding modules
// embedwt - handle options depending on whether pre-trained weights supplied
// embed, embedbag - process args and return Embedding/EmbeddingBag module
// --------------------------------------------------------------------------------------
void embedset(Cast,Setting,const Pairs&,torch::nn::EmbeddingOptions&);
void embedset(Cast,Setting,const Pairs&,torch::nn::EmbeddingBagOptions&);

template<typename O> void embedpair(Cast c,Pairs& p,O& o,Tensor& w,bool &z) {
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::rows:       o.num_embeddings(int64(p,c)); break;
   case Setting::cols:       o.embedding_dim (int64(p,c)); break;
   case Setting::padindex:   o.padding_idx(int64n(p,c)); break;
   case Setting::maxnorm:    o.max_norm(optdouble(p,c)); break;
   case Setting::p:          o.norm_type(mdouble(p,c)); break;
   case Setting::scale:      o.scale_grad_by_freq(mbool(p,c)); break;
   case Setting::sparse:     o.sparse(mbool(p,c)); break;
   case Setting::weight:     if(!pempty(p)) pten(p,w); break;
   case Setting::freeze:     z=mbool(p,c); break;
   case Setting::mode:       embedset(c,Setting::mode,p,o); break;
   case Setting::lastoffset: embedset(c,Setting::lastoffset,p,o); break;
   default: mpair(c,p); break;
  }
}

template<typename M,typename O> M embedwt(Cast c,O o,const Tensor& w,bool z) {
 bool a=o.num_embeddings()!=nj,b=o.embedding_dim()!=nj;
 if(w.defined()) {
  TORCH_CHECK(w.dim()==2, msym(c),": ",w.dim(),"-dim weights given, 2-dim matrix expected");
  TORCH_CHECK(w.is_floating_point(), msym(c),": weight matrix is not floating point");
  if(!a) o.num_embeddings(w.size(0));
  else TORCH_CHECK(o.num_embeddings()==w.size(0), "rows = ",o.num_embeddings()," but weights are ",w.sizes());
  if(!b) o.embedding_dim(w.size(1));
  else TORCH_CHECK(o.embedding_dim() ==w.size(1), "cols = ",o.embedding_dim(), " but weights are ",w.sizes());
   M m=M(o._weight(w));
   m->weight.set_requires_grad(!z);
   return m;
 } else {
  TORCH_CHECK(a,"embed: supply number of rows in the embedding matrix");
  TORCH_CHECK(b,"embed: supply number of cols in the embedding matrix");
  return M(o);
 }
}

torch::nn::Embedding embed(K,J,Cast);
torch::nn::EmbeddingBag embedbag(K,J,Cast);

// ----------------------------------------------------------------------------
// retrieve settings from existing Embedding/EmbeddingBag:
// embedget - functions to retrieve options specific to Embedding/EmbeddingBag
// embed - templated function which gets options and initial optional weights
// ----------------------------------------------------------------------------
void embedget(bool,bool,K,Cast,Setting,const torch::nn::EmbeddingOptions&,const torch::nn::EmbeddingOptions&);
void embedget(bool,bool,K,Cast,Setting,const torch::nn::EmbeddingBagOptions&,const torch::nn::EmbeddingBagOptions&);

template<typename O> K embed(bool a,Cast c,const O& o,const Tensor& w) {
 K x=KDICT; O d(o.num_embeddings(),o.embedding_dim());
 if(o._weight().defined()) {
  msetting(x, Setting::weight, kget(o._weight()));
  msetting(x, Setting::freeze, kb(!w.requires_grad()));
 } else {
  msetting(x, Setting::rows, kj(o.num_embeddings()));
  msetting(x, Setting::cols, kj(o.embedding_dim()));
 }
 embedget(a,false,x,c,Setting::padindex,o,d);   // embedding only
 if(a || o.max_norm().has_value())                         msetting(x, Setting::maxnorm, kf(o.max_norm() ? o.max_norm().value() : nf));
 if(a || o.norm_type()          != d.norm_type())          msetting(x, Setting::p,       kf(o.norm_type()));
 if(a || o.scale_grad_by_freq() != d.scale_grad_by_freq()) msetting(x, Setting::scale,   kb(o.scale_grad_by_freq()));
 embedget(a,true,x,c,Setting::mode,o,d);        // embedding bag only
 if(a || o.sparse()             != d.sparse())             msetting(x, Setting::sparse,  kb(o.sparse()));
 embedget(a,true,x,c,Setting::lastoffset,o,d);  // embedding bag only
 embedget(a,true,x,c,Setting::padindex,o,d);    // embedding bag only
 return x;
}

} // knn namespace

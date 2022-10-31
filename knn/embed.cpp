#include "../ktorch.h"
#include "embed.h"

namespace knn {

// -----------------------------------------------------------------------------
// EmbedPosition - learn position embedding given max sequence length, embed dim
// -----------------------------------------------------------------------------
EmbedPositionImpl::EmbedPositionImpl(const EmbedPositionOptions& o) : options(o) {
 reset();
}

void EmbedPositionImpl::reset() {
 pos = register_parameter("pos", torch::zeros({1,options.rows(),options.cols()}));
}

void EmbedPositionImpl::pretty_print(std::ostream& s) const {
 s << "knn::EmbedPosition(rows=" << options.rows() << ", cols=" << options.cols() << ")";
}

Tensor EmbedPositionImpl::forward(const Tensor& x) {
 using namespace torch::indexing;
 return pos.index({Slice(), Slice(None,x.size(-1)), Slice()});
}

// -----------------------------------------------------
// embedpos - set/get options for position embedding
// -----------------------------------------------------
EmbedPositionOptions embedpos(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); EmbedPositionOptions o(0,0);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.rows(int64(x,i+j,c,Setting::rows)); break;
    case 1: o.cols(int64(x,i+j,c,Setting::cols)); break;
    default: mpos(x,c,i+j); break;
   }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::rows: o.rows(int64(p,c)); break;
   case Setting::cols: o.cols(int64(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.rows()>0, msym(c), ": positive number of rows required");
 TORCH_CHECK(o.cols()>0, msym(c), ": positive number of columns required");
 return o;
}

K embedpos(bool a,const EmbedPositionOptions& o) {
 K x=KDICT;
 msetting(x, Setting::rows, kj(o.rows()));
 msetting(x, Setting::cols, kj(o.cols()));
 return resolvedict(x);
}

// -----------------------------------------------------------------------------
// EmbedSequence - learn token & position embedding given rows,cols,length
//                 rows - vocabulary size for tokens
//                 cols - embedding dimension (attributes for each token)
//                 length - maximum sequence length
// -----------------------------------------------------------------------------
EmbedSequenceImpl::EmbedSequenceImpl(const EmbedSequenceOptions& o) : options(o) {
 reset();
}

void EmbedSequenceImpl::reset() {
 tok = register_module("tok", torch::nn::Embedding(options.rows(),options.cols()));
 pos = register_module("pos",   knn::EmbedPosition(options.length(),options.cols()));
}

void EmbedSequenceImpl::pretty_print(std::ostream& s) const {
 s << "knn::EmbedSequence(rows=" << options.rows() << ", cols=" << options.cols() << ", length=" << options.length() <<  ")";
}

Tensor EmbedSequenceImpl::forward(const Tensor& x) {
 return tok(x) + pos(x);
}

// ---------------------------------------------------------
// embedseq - set/get options for token & position embedding
// ---------------------------------------------------------
EmbedSequenceOptions embedseq(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); EmbedSequenceOptions o(0,0,0);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0:   o.rows(int64(x,i+j,c,Setting::rows)); break;
    case 1:   o.cols(int64(x,i+j,c,Setting::cols)); break;
    case 2: o.length(int64(x,i+j,c,Setting::length)); break;
    default: mpos(x,c,i+j); break;
   }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::rows:   o.rows(int64(p,c)); break;
   case Setting::cols:   o.cols(int64(p,c)); break;
   case Setting::length: o.length(int64(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.rows()>0, msym(c), ": positive number of rows required for token embedding");
 TORCH_CHECK(o.cols()>0, msym(c), ": positive number of columns required for token & position embedding");
 TORCH_CHECK(o.length()>0, msym(c), ": positive sequence length required for position embedding");
 return o;
}

K embedseq(bool a,const EmbedSequenceOptions& o) {
 K x=KDICT;
 msetting(x, Setting::rows,   kj(o.rows()));
 msetting(x, Setting::cols,   kj(o.cols()));
 msetting(x, Setting::length, kj(o.length()));
 return resolvedict(x);
}

// --------------------------------------------------------------------------------------
// create embedding/embedding bag module given options:
// embedmode - translate symbol to internal embedding mode (variant)
// embedset - set name/value pairs specific to Embedding vs EmbeddingBag
// embed, embedbag - process args and return Embedding/EmbeddingBag module
// --------------------------------------------------------------------------------------
static torch::nn::EmbeddingBagMode embedmode(S s) {
 switch(emap(s)) {
  case Enum::sum:  return torch::kSum;
  case Enum::mean: return torch::kMean;
  case Enum::max:  return torch::kMax;
  default: TORCH_ERROR("unrecognized mode for embedding bag: ",s);
 }
}

void embedset(Cast c,Setting s,const Pairs& p,torch::nn::EmbeddingOptions& o) {
 TORCH_ERROR("unrecognized option for ",msym(c),": ",mset(s));
}

void embedset(Cast c,Setting s,const Pairs& p,torch::nn::EmbeddingBagOptions& o) {
 if     (s == Setting::mode)       o.mode(embedmode(code(p,c)));
 else if(s == Setting::lastoffset) o.include_last_offset(mbool(p,c));
 else TORCH_ERROR("unrecognized option for ",msym(c),": ",mset(s));
}

torch::nn::Embedding embed(K x,J i,Cast c) {
 bool z=false; Pairs p; Tensor w; J n=xargc(x,i,p);
 torch::nn::EmbeddingOptions o(nj,nj);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0:
     switch(kK(x)[i+j]->t) {
      case 0:   if(!xten(x,i+j,w)) w=kput(x,i+j); break;
      case -KJ: o.num_embeddings(int64(x,i+j,c,Setting::rows)); break;
      default:  TORCH_ERROR("embed: 1st arg is number of rows of weight matrix");
     }
     break;
    case 1: 
     if(w.defined()) z=mbool(x,i+j,c,Setting::freeze);
     else  o.embedding_dim(int64(x,i+j,c,Setting::cols));
     break;
    case 2: o.padding_idx(int64n(x,i+j,c,Setting::padindex)); break;
    case 3: o.max_norm(optdouble(x,i+j,c,Setting::maxnorm)); break;
    case 4: o.norm_type(mdouble(x,i+j,c,Setting::p)); break;
    case 5: o.scale_grad_by_freq(mbool(x,i+j,c,Setting::scale)); break;
    case 6: o.sparse(mbool(x,i+j,c,Setting::sparse)); break;
    default: mpos(x,c,i+j); break;
   }
 }
 embedpair(c,p,o,w,z);
 return embedwt<torch::nn::Embedding,torch::nn::EmbeddingOptions>(c,o,w,z);
}

torch::nn::EmbeddingBag embedbag(K x,J i,Cast c) {
 bool z=false; Pairs p; Tensor w; J n=xargc(x,i,p);
 torch::nn::EmbeddingBagOptions o(nj,nj);
 // allow mode if last arg even if early in sequence
 if(!x->t && n>1 && n<6 && xsym(x,i+n-1))
  n--, o.mode(embedmode(code(x,i+n,c,Setting::mode)));
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0:
     switch(kK(x)[i+j]->t) {
      case 0:   if(!xten(x,i+j,w)) w=kput(x,i+j); break;
      case -KJ: o.num_embeddings(int64(x,i+j,c,Setting::rows)); break;
      default:  TORCH_ERROR("embed: 1st arg is number of rows or weight matrix");
     }
     break;
    case 1: 
     if(w.defined()) z=mbool(x,i+j,c,Setting::freeze);
     else  o.embedding_dim(int64(x,i+j,c,Setting::cols));
     break;
    case 2: o.max_norm(optdouble(x,i+j,c,Setting::maxnorm)); break;
    case 3: o.norm_type(mdouble(x,i+j,c,Setting::p)); break;
    case 4: o.scale_grad_by_freq(mbool(x,i+j,c,Setting::scale)); break;
    case 5: o.mode(embedmode(code(x,i+j,c,Setting::mode))); break;
    case 6: o.sparse(mbool(x,i+j,c,Setting::sparse)); break;
    case 7: o.padding_idx(int64n(x,i+j,c,Setting::padindex)); break;
    default: mpos(x,c,i+j); break;
   }
 }
 embedpair(c,p,o,w,z);
 return embedwt<torch::nn::EmbeddingBag,torch::nn::EmbeddingBagOptions>(c,o,w,z);
}

// ---------------------------------------------------------------------------
// retrieve settings from existing Embedding/EmbeddingBag:
// embedget - functions to retrieve options specific to Embedding/EmbeddingBag
// ---------------------------------------------------------------------------
void embedget(bool a,bool b,K x,Cast c,Setting s,const torch::nn::EmbeddingOptions& o,const torch::nn::EmbeddingOptions& d) {
 if(!b && s == Setting::padindex && (a || o.padding_idx().has_value()))
  msetting(x, s, kj(o.padding_idx() ? o.padding_idx().value() : nj));
}

void embedget(bool a,bool b,K x,Cast c,Setting s,const torch::nn::EmbeddingBagOptions& o,const torch::nn::EmbeddingBagOptions& d) {
 if(s == Setting::mode && (a || o.mode().index() != d.mode().index()))
  msetting(x, s, ks(ESYM(o.mode())));
 else if(s == Setting::lastoffset && (a || o.include_last_offset() != d.include_last_offset()))
  msetting(x, s, kb(o.include_last_offset()));
 else if(b && s == Setting::padindex && (a || o.padding_idx().has_value()))
  msetting(x, s, kj(o.padding_idx() ? o.padding_idx().value() : nj));
}

} // knn namespace

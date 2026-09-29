#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <utility>
namespace avi {
// Greedy target verification: every emitted token is selected by the target.
// EOS/budget termination must be checked even when a proposal is accepted.
struct GreedyVerification {
  std::vector<int64_t> proposals,eos,tokens;
  int budget,accepted=0;
  bool done=false;
  GreedyVerification(std::vector<int64_t> draft,int limit,std::vector<int64_t> stops)
      :proposals(std::move(draft)),eos(std::move(stops)),budget(limit){
    if(limit<1)throw std::invalid_argument("Empty speculative output budget");
  }
  void observe(int64_t target){
    if(done)throw std::logic_error("Verification already complete");
    size_t index=tokens.size();tokens.push_back(target);
    bool match=index<proposals.size()&&target==proposals[index];
    if(match)++accepted;
    done=!match||int(tokens.size())>=budget||std::find(eos.begin(),eos.end(),target)!=eos.end();
  }
  int committed_inputs() const {return accepted+1;}
};
}

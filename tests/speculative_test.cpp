#include "avi/speculative.h"
#include <iostream>
#include <stdexcept>
static void check(bool value){if(!value)throw std::runtime_error("Speculative decision regression");}
int main(){
  using avi::GreedyVerification;
  {GreedyVerification d({2,3,4},4,{9});d.observe(7);check(d.done&&d.accepted==0&&d.committed_inputs()==1&&d.tokens==std::vector<int64_t>{7});}
  {GreedyVerification d({2,3,4},4,{9});d.observe(2);d.observe(7);check(d.done&&d.accepted==1&&d.committed_inputs()==2);}
  {GreedyVerification d({2,3,4},4,{9});d.observe(2);d.observe(3);d.observe(4);check(!d.done);d.observe(8);check(d.done&&d.accepted==3&&d.committed_inputs()==4);}
  {GreedyVerification d({2,9,4},4,{9});d.observe(2);d.observe(9);check(d.done&&d.accepted==2&&d.tokens.size()==2);}
  {GreedyVerification d({2,3,4},4,{9});d.observe(2);d.observe(9);check(d.done&&d.accepted==1&&d.committed_inputs()==2);}
  {GreedyVerification d({},1,{9});d.observe(3);check(d.done&&d.tokens.size()==1&&d.committed_inputs()==1);}
  {GreedyVerification d({2,3},1,{});d.observe(2);check(d.done&&d.tokens.size()==1);}
  // Exhaust every possible rejection position for windows 1..5.
  for(int k=1;k<=5;++k)for(int accepted=0;accepted<=k;++accepted){
    std::vector<int64_t> proposals(k,2);GreedyVerification d(proposals,k+1,{9});
    for(int i=0;i<accepted;++i)d.observe(2);
    d.observe(3);check(d.done&&d.accepted==accepted&&int(d.tokens.size())==accepted+1);
  }
  std::cout<<"Greedy MTP rejection, acceptance, bonus, EOS and budgets passed\n";
}

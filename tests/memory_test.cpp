#include "avi/memory_budget.h"
#include <iostream>
#include <fstream>
static void check(bool ok){if(!ok)throw std::runtime_error("Budget invariant failed");}
int main(int argc,char** argv){try{
 avi::MemoryBudget b(100);check(b.reserve(1,60));check(!b.reserve(2,41));check(b.used()==60);check(b.reserve(2,40));check(!b.fits(1));b.release(1);check(b.fits(60));b.release(2);check(b.used()==0);
 bool error=false;try{b.release(3);}catch(const std::logic_error&){error=true;}check(error);
 error=false;try{avi::checked_product(UINT64_MAX,2);}catch(const std::overflow_error&){error=true;}check(error);
 if(argc>1){std::ifstream f(argv[1]);nlohmann::json config;f>>config;auto a=avi::session_bytes(config,2,100),c=avi::session_bytes(config,2,200);check(c>a);check(avi::session_bytes(config,4,100)<a);}
 std::cout<<"Memory budget tests passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}

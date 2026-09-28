#include "avi/json_grammar.h"
#include <iostream>
#include <stdexcept>
static void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(){try{
 for(auto text:{"{}"," {\"a\":[true,false,null,-0.25e+2,{\"b\":\"x\\u4e2d\"}]} ","{\"name\":\"中文\"}"}){
  avi::JsonGrammar g;for(auto c:std::string(text))check(g.feed(std::string(1,c)),"Valid incremental JSON rejected");check(g.complete(),"Valid JSON incomplete");}
 for(auto text:{"[]","{a:1}","{\"a\":01}","{\"a\":1,}","{\"a\":[1,]}","{\"a\":NaN}","{\"a\":1.}","{}{}","{\"a\":\"x\n\"}"}){avi::JsonGrammar g;check(!g.feed(text),"Invalid JSON accepted");}
 avi::JsonGrammar partial;check(partial.feed("{\"a\":\"\\u4"),"Valid prefix rejected");check(!partial.complete(),"Partial JSON accepted as complete");
 avi::JsonGrammar utf;check(!utf.feed(std::string("{\"a\":\"")+char(0xc0)),"Invalid UTF8 accepted");
 std::cout<<"JSON grammar tests passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
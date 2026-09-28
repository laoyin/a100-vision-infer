#pragma once
#include <string>
#include <vector>
#include <cstdint>
namespace avi {
// Incremental JSON-object grammar. No accumulated string contents: states can share vocabulary masks.
class JsonGrammar {
 struct Frame {char kind;int expect;}; // O: 0 key/end,1 colon,2 value,3 comma/end,4 key; A:5 value/end,6 comma/end,7 value
 std::vector<Frame> stack_;char lex_=0;int number_=0,literal_at_=0,unicode_=0,utf_=0,lo_=128,hi_=191;
 std::string literal_;bool escape_=false,key_=false,started_=false,done_=false;
 bool value_done(){if(stack_.empty()){done_=true;return true;}auto& f=stack_.back();if(f.kind=='O'&&f.expect==2)f.expect=3;else if(f.kind=='A'&&(f.expect==5||f.expect==7))f.expect=6;else return false;return true;}
 bool close(char kind){if(stack_.empty()||stack_.back().kind!=kind)return false;stack_.pop_back();return value_done();}
 bool start(unsigned char c){if(c=='{'||c=='['){if(stack_.size()>=64)return false;stack_.push_back({c=='{'?'O':'A',c=='{'?0:5});return true;}if(c=='"'){lex_='s';key_=false;return true;}if(c=='-'||(c>='0'&&c<='9')){lex_='n';number_=c=='-'?0:(c=='0'?1:2);return true;}if(c=='t'||c=='f'||c=='n'){lex_='l';literal_=c=='t'?"true":(c=='f'?"false":"null");literal_at_=1;return true;}return false;}
 bool character(unsigned char c){
  if(lex_=='s'){
   if(utf_){if(c<lo_||c>hi_)return false;utf_--;lo_=128;hi_=191;return true;}
   if(unicode_){if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F')))return false;unicode_--;return true;}
   if(escape_){escape_=false;if(c=='u'){unicode_=4;return true;}return c=='"'||c=='\\'||c=='/'||c=='b'||c=='f'||c=='n'||c=='r'||c=='t';}
   if(c=='\\'){escape_=true;return true;}if(c=='"'){lex_=0;if(key_){stack_.back().expect=1;return true;}return value_done();}if(c<32)return false;
   if(c<128)return true;if(c>=194&&c<=223){utf_=1;return true;}if(c>=224&&c<=239){utf_=2;lo_=c==224?160:128;hi_=c==237?159:191;return true;}if(c>=240&&c<=244){utf_=3;lo_=c==240?144:128;hi_=c==244?143:191;return true;}return false;
  }
  if(lex_=='l'){if(c!=static_cast<unsigned char>(literal_[literal_at_]))return false;if(++literal_at_==int(literal_.size())){lex_=0;literal_.clear();literal_at_=0;return value_done();}return true;}
  if(lex_=='n'){
   bool digit=c>='0'&&c<='9';
   if(number_==0){if(!digit)return false;number_=c=='0'?1:2;return true;}
   if(number_==1||number_==2){if(digit){if(number_==1)return false;return true;}if(c=='.'){number_=3;return true;}if(c=='e'||c=='E'){number_=5;return true;}}
   else if(number_==3){if(!digit)return false;number_=4;return true;}
   else if(number_==4){if(digit)return true;if(c=='e'||c=='E'){number_=5;return true;}}
   else if(number_==5){if(c=='+'||c=='-'){number_=6;return true;}if(!digit)return false;number_=7;return true;}
   else if(number_==6){if(!digit)return false;number_=7;return true;}
   else if(number_==7&&digit)return true;
   if(!(c==' '||c=='\t'||c=='\r'||c=='\n'||c==','||c=='}'||c==']'))return false;
   lex_=0;number_=0;if(!value_done())return false;return character(c);
  }
  if(c==' '||c=='\t'||c=='\r'||c=='\n')return true;if(done_)return false;
  if(!started_){if(c!='{')return false;started_=true;stack_.push_back({'O',0});return true;}
  if(stack_.empty())return false;auto& frame=stack_.back();
  if(frame.kind=='O'){
   if(frame.expect==0||frame.expect==4){if(c=='}'&&frame.expect==0)return close('O');if(c!='"')return false;lex_='s';key_=true;return true;}
   if(frame.expect==1){if(c!=':')return false;frame.expect=2;return true;}
   if(frame.expect==2)return start(c);
   if(c=='}')return close('O');if(c==','){frame.expect=4;return true;}return false;
  }
  if(frame.expect==5||frame.expect==7){if(c==']'&&frame.expect==5)return close('A');return start(c);}
  if(c==']')return close('A');if(c==','){frame.expect=7;return true;}return false;
 }
 public:
 bool feed(const std::string& bytes){for(unsigned char c:bytes)if(!character(c))return false;return true;}
 bool complete() const{return done_&&lex_==0&&stack_.empty();}
 std::string signature() const{std::string s;s.push_back(lex_);s.push_back(number_);s.push_back(literal_at_);s+=literal_;s.push_back(0);for(int n:{unicode_,utf_,lo_,hi_,int(escape_),int(key_),int(started_),int(done_)})s.push_back(char(n));for(auto f:stack_){s.push_back(f.kind);s.push_back(char(f.expect));}return s;}
};
}
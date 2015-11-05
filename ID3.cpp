#include<iostream>
#include<fstream>
#include<string>
#include<set>
#include<unordered_map>
#include<algorithm>
#include<vector>
#include<utility>
#include<cmath>
#include<iomanip>
using namespace std;

ofstream fout("idrtree.out");

struct node{
	vector<string> attributes;
	string split_attribute;
	int split_attr_idx;
	vector<string> attribute_median;
	vector<double> purity_value; //corespunzator cu atributele

	vector<int> instance_id;

	vector< pair<string,node*> > sons;

	string decision;

	node() {

		decision="";
		attributes.clear();
		attribute_median.clear();
		split_attribute="";
		purity_value.clear();
		instance_id.clear();
		sons.clear();

	}
};

unordered_map<int, unordered_map<string,string> > training_info;
unordered_map<string,string> test_info;

vector<string> attributes;
string decision_attribute;

double entropy(vector<string> &some_values) {

	double e=0;

	//group same values
	unordered_map<string,int> group_vals;

	for (int i=0; i<some_values.size(); ++i)
	 group_vals[some_values[i]]++;

	unordered_map<string,int>::iterator it;

	for (it=group_vals.begin(); it!=group_vals.end(); ++it) {
		double prob=(double)it->second/(double)some_values.size();
		e+=(prob*log2(1.0/prob));
	}

	return e;
}

double purity_function(vector<string> &attribute_values, vector<string> &decision_values) {

	double rez=entropy(attribute_values);

	//break curent attributes over decision values
	unordered_map<string, vector<string> > group_attr;

	for (int i=0; i<decision_values.size(); ++i)
	 group_attr[decision_values[i]].push_back(attribute_values[i]);

	unordered_map<string, vector<string> >::iterator it;

	for (it=group_attr.begin(); it!=group_attr.end(); ++it)
	 rez-=( entropy(it->second)*(double)it->second.size()/(double)attribute_values.size() );

	return rez;
}

node *build_id3(node *curent_node) {

	//1 -- verify if it is a decision node => all decisions are the same for all instances
	bool same=1;
	string first=training_info[curent_node->instance_id[0]][decision_attribute];

	for (int j=1; j<curent_node->instance_id.size(); ++j)
		 if (training_info[curent_node->instance_id[j]][decision_attribute]!=first) {
		   same=0;
		   break;
		 }

	if (same==1) {
	  curent_node->decision=first;
	  return curent_node;
	}
	
	if (curent_node->attributes.empty()) {

	   //find maximum decision value
	   unordered_map<string,int> max_decision;
	   int maxim=0;
	   string decision="";
	   
	   for (int i=0; i<curent_node->instance_id.size(); ++i){
		int aux=++max_decision[training_info[curent_node->instance_id[i]][decision_attribute]];
		if (aux>=maxim) { maxim=aux; decision=training_info[curent_node->instance_id[i]][decision_attribute]; }
	   }
		
	   curent_node->decision=decision;
	   return curent_node;
	}

	//3 -- calculate purity function (info_gain) for all curent attributes
	vector<string> attribute_values;
	vector<string> decision_values;

	for (int j=0; j<curent_node->instance_id.size(); ++j)
     decision_values.push_back(training_info[curent_node->instance_id[j]][decision_attribute]);

	for (int j=0; j<curent_node->attributes.size(); ++j) {

		attribute_values.clear();

		for (int i=0; i<curent_node->instance_id.size(); ++i)
		attribute_values.push_back(training_info[curent_node->instance_id[i]][curent_node->attributes[j]]);
		
		//normalize atribute values in 0s and 1ns and register median
		sort(attribute_values.begin(),attribute_values.end());
		curent_node->attribute_median.push_back(attribute_values[attribute_values.size()/2]);
		
		for (int i=0; i<attribute_values.size(); ++i)
		 if (attribute_values[i]<=curent_node->attribute_median[j]) attribute_values[i]='0';
		 else attribute_values[i]='1';

		curent_node->purity_value.push_back( purity_function(attribute_values,decision_values) );
	}

	//4 -- find the best attribute and set split attribute of curent node
    double maxim_ig=curent_node->purity_value[0];
	int idx=0;

	for (int i=1; i<curent_node->purity_value.size(); ++i)
	 if (curent_node->purity_value[i]>maxim_ig) {
		 maxim_ig=curent_node->purity_value[i];
		 idx=i;
	 }

    curent_node->split_attribute=curent_node->attributes[idx];
    curent_node->split_attr_idx=idx;
    
    if (maxim_ig<0.6) return curent_node;

    //5 -- split over the idx atribute and make sons
    unordered_map<string, vector<int> > group_for_split;

    for (int i=0; i<curent_node->instance_id.size(); ++i){
	 string side="0";
	 if (training_info[curent_node->instance_id[i]][curent_node->split_attribute]>curent_node->attribute_median[curent_node->split_attr_idx]) side="1";
	 group_for_split[side].push_back(curent_node->instance_id[i]);
    }

	unordered_map<string, vector<int> >::iterator it;

	for (it=group_for_split.begin(); it!=group_for_split.end(); ++it) {

		node* curent_son=new node();
		//populate initial info for curent son

		  //-- set sons attribute list all except idx
		  for (int i=0; i<curent_node->attributes.size(); ++i)
		   if (i!=idx) curent_son->attributes.push_back(curent_node->attributes[i]);
		  //-- set sons id list
		  for (int i=0; i<it->second.size(); ++i)
		   curent_son->instance_id.push_back(it->second[i]);
		  //build all info for curent son
		  curent_son=build_id3(curent_son);
		//-------
		curent_node->sons.push_back(make_pair(it->first,curent_son));
	}

	return curent_node;
}

string find_decision(node *curent_node) {

       if (curent_node->decision.length()>0)
	    return curent_node->decision;

       string test_split_value=test_info[curent_node->split_attribute];
       
       string median=curent_node->attribute_median[curent_node->split_attr_idx];
       
       if (test_split_value<=median) test_split_value='0';
       else test_split_value='1';

       bool ok=0;

       for (int i=0; i<curent_node->sons.size(); ++i)
        if (curent_node->sons[i].first==test_split_value) {
                                                           ok=1;
                                                           return find_decision(curent_node->sons[i].second);
     	                                                   }

      if ( !ok ) {
      	
      	 //find maximum decision value
	   unordered_map<string,int> max_decision;
	   int maxim=0;
	   string decision="";

	   for (int i=0; i<curent_node->instance_id.size(); ++i){
		int aux=++max_decision[training_info[curent_node->instance_id[i]][decision_attribute]];
		if (aux>=maxim) { maxim=aux; decision=training_info[curent_node->instance_id[i]][decision_attribute]; }
	   }

	   return decision;
      	
      }
}

void print_tree(node *curent_node,string spaces) {

	 //print information about curent node
	 fout<<spaces<<curent_node->split_attribute<<"("<<curent_node->instance_id.size()<<")"<<"\n";
	 
	 for (int i=0; i<curent_node->sons.size(); ++i)
	   print_tree(curent_node->sons[i].second,spaces+"  ");
	   
	 fout<<spaces<<curent_node->split_attribute<<"-->End subtree"<<"\n";
	   
}

int main(void) {

	ifstream cin("covtype-train.csv");

	//read attributes and decision type
	string s;
	getline(cin,s);

	string aux="";

	for (int i=0; i<s.length(); ++i) {

		if (s[i]==',') {
		   attributes.push_back(aux);
		   aux="";
	    }
		else aux+=s[i];

	}

	attributes.push_back(aux);//last atribute is decision
	decision_attribute=aux;

	//read info about instances
	int iid=1;
	while (getline(cin,s)) {

		aux="";
		int attrnum=0;

		for (int i=0; i<s.length(); ++i)
		   if (s[i]==',') {
		       training_info[iid][attributes[attrnum]]=aux;
		       ++attrnum;
		       aux="";
	       }
		   else aux+=s[i];

		training_info[iid][attributes[attrnum]]=aux;

		++iid;
    }

    //build root node
    node *root=new node();

    for (int i=0; i<attributes.size()-1; ++i) //no decision atribute in nodes
	 root->attributes.push_back(attributes[i]);

	for (int i=1; i<iid; ++i)
	 root->instance_id.push_back(i);

	//build the id3-tree
	root=build_id3(root);

	//read test instances and count corect results
	ifstream fin("covtype-test.csv");

	vector<string> test_attributes;

	getline(fin,s);

	aux="";

	for (int i=0; i<s.length(); ++i) {

		if (s[i]==',') {
		   test_attributes.push_back(aux);
		   aux="";
	    }
		else aux+=s[i];
	}

	//read instances
	int numberOfTests=0;
	int goodAnswers=0;
	int unknown=0;

	while (getline(fin,s)) {

          ++numberOfTests;
          aux="";
          test_info.clear();
          int attrnum=0;

          for (int i=0; i<s.length(); ++i)
          if (s[i]==',') {
                         test_info[test_attributes[attrnum]]=aux;
                         aux="";
                         ++attrnum;
                         }
          else aux+=s[i];

          //in aux i have the right cover type
          string decision_for_test= "";
		  decision_for_test=find_decision(root);
		  
          if (aux==decision_for_test) ++goodAnswers;
          else if (decision_for_test=="?") ++unknown;

		  }

    cout<<"Total number of tests: "<<numberOfTests<<"\n";
    cout<<"Good Answers: "<<goodAnswers<<"\n";
    cout<<"Unknown Answers: "<<unknown<<"\n";
    cout<<"Guess Percent: "<<setprecision(3)<<fixed<<100.0*(double)goodAnswers/(double)numberOfTests;
    
    //print the tree
    fout<<"\n\n";
    print_tree(root,"");

	return 0;
}

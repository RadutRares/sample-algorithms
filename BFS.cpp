#include <list>
#include <map>
#include <queue>
#include <iostream>
#include <fstream>

class associationList {
	std::map<char, std::list <char>> alist;
public:
	void add_edge(char a, char b) {
		alist[a].emplace_back(b);
	}
	std::list <char> get_vertexes(char a){
		return alist[a];
	}
};


int main (int argc, char * argv[]) {
	associationList list;
	char search;// Target 

	if (argc > 1) {
		const std::string& file = argv[1];
		std::ifstream in {file};
		int edges;
		char a, b;
		in >> edges;
		for (int i = 0; i < edges; ++i){
			in >> a;
			in >> b;
			list.add_edge(a,b);
		}
		in >> search;
	} else {
		std::cout << "please provide a file" << std::endl; 
	}
	
	// BFS
	char start = 'a'; // Starting at 'a'
	std::queue<char> toExplore;
	std::map<char,bool> explored;
 
	toExplore.push(start);

	while (!toExplore.empty()){
		char current = toExplore.front();
		toExplore.pop();
		std::list <char> neighbours = list.get_vertexes(current); 

		for(auto elem:neighbours){
			std::string e = explored[elem] ? "True" : "False";
			std::cout << elem << " : "<< e << std::endl;
			if (!explored[elem]){
				toExplore.push(elem);
			}
		}
		explored[current] = true;
	}

	for (auto elem: explored){
		std::string e = elem.second ? "True" : "False";
		std::cout << "Found " << elem.first << " " << e << std::endl;
	}
}




// This C++ script takes an fml and checks if its subfmls are under the restricted length
// Assuming the fml is flat (non-branching operators are always under branching operators)
// Assuming the input is in Lisp-like string notation

#include "utils.h"
#include <iostream>
#include <cassert>

void extractsubHs(string& fml, vector<string>& subHs) {
	// get the head and rest of fml
	string head = getHead(fml);
	vector<string> rest = getRest(fml);

	// function body
	if (head == "max" || head == "min") {
		extractsubHs(rest[0], subHs);
		extractsubHs(rest[1], subHs);
	} else if (head == "if") {
		extractsubHs(rest[1], subHs);
		extractsubHs(rest[2], subHs);
	} else subHs.push_back(fml);
}

bool checkHelper(vector<string>& subHs, i64& restrLen) {
	// Base cases
	if (!subHs.size()) return true;
	else if (fmlSize(subHs[0]) > restrLen) return false;

	// Inductive case
	subHs.erase(subHs.begin()); // remove the first element
	return checkHelper(subHs, restrLen);
}

bool checkRestrOnSubfmls(string& fml, i64& restrLen) {
	// extract the subheuristic formulae from fml
	vector<string> subHs = {};
	extractsubHs(fml, subHs);

	// return if subHs respect the restrictions
	return checkHelper(subHs, restrLen);
}

extern "C" bool extCheckRestrOnSubfmls(char*& fmlRaw, i64& restrLen) {
	string fml = fmlRaw;
	bool flag = checkRestrOnSubfmls(fml, restrLen);
	return flag;
}


// This C++ script takes a string function and its inputs and evaluates it
// Assuming the input is in Lisp-like string notation

#include "utils.h"
#include <iostream>
#include <cassert>

//double evaluate(string& funcStr, double& x, double& y, double& t);
double evaluate(string& funcStr, double& t);

//bool evalCond(string& condStr, double& x, double& y, double& t) {
bool evalCond(string& condStr, double& t) {
	// get the first and rest of the function string
	string head = getHead(condStr);
	vector<string> rest = getRest(condStr);

	// Base Case:
	// evaluate the conditions
	if (head == "==") {
		double left = evaluate(rest[0], t);
		double right = evaluate(rest[1], t);
		if (left == right) return true;
		else return false;

	}
	if (head == ">") {
		double left = evaluate(rest[0], t);
		double right = evaluate(rest[1], t);
		if (left > right) return true;
		else return false;
	}
	if (head == "<") {
		double left = evaluate(rest[0], t);
		double right = evaluate(rest[1], t);
		if (left < right) return true;
		else return false;
	}
	if (head == ">=") {
		double left = evaluate(rest[0], t);
		double right = evaluate(rest[1], t);
		if (left >= right) return true;
		else return false;
	}
	if (head == "<=") {
		double left = evaluate(rest[0], t);
		double right = evaluate(rest[1], t);
		if (left <= right) return true;
		else return false;
	}

	// Inductive Case:
	// there can be arbitrary number of subconditionals
	if (head == "&&") {
		for (double i = 0; i < rest.size(); i++) {
			bool curcond = evalCond(rest[i], t);
			if (!curcond) return false;
		}
		return true;
	}
	if (head == "||") {
		for (double i = 0; i < rest.size(); i++) {
			bool curcond = evalCond(rest[i], t);
			if (curcond) return true;
		}
		return false;
	}

	// default
	return false;
}

double handleCondArgs(vector<string>& args, double& t) {
	// handles the arguments (a.k.a. the args part) of the cond form: (cond args)
	// assuming args contains <"((cond_1) case_1)", "((cond_2) case_2)", ...>
	for (double i = 0; i < args.size(); i++) {
		// Take the cond and the case strings
		string cond = getHead(args[i]);
		vector<string> casevec = getRest(args[i]); // a vec of 1 element

		// see if the cond evaluates to true
		if (evalCond(cond, t) || cond == "else") return evaluate(casevec[0], t);
	}

	// The default case
	string cond = getHead(args[args.size() - 1]);
	vector<string> casevec = getRest(args[args.size() - 1]); // a vec of 1 element
	return evaluate(casevec[0], t);
}

//double evaluate(string& funcStr, double& x, double& y, double& t) {
double evaluate(string& funcStr, double& t) {
	double x = 0;
	double y = 0;
	// get the first and rest of the function string
	string head = getHead(funcStr);
	vector<string> rest = getRest(funcStr);

	// Base Case: funcStr is terminal
	// a constant
	if (isdigit(head[0]) || head[0] == '.') return stod(head);


	// Special Case: need to handle cond
	// the rest is of the form <"((cond_1) case_1)", "((cond_2) case_2)", ...>	
	if (head == "cond") return handleCondArgs(rest, t);

	// variable
	if (rest.size() == 0) {
		if (head == "x") return x;
		if (head == "y") return y;
		if (head == "t") return t;
	}

	// Inductive Case: needs recursions
	// Unary operators
	if (rest.size() == 1) {
		// Compute the subtree
		double subtree = evaluate(rest[0], t);

		// Compute using unary operators
		if (head == "sqrt") return sqrt(subtree);
		if (head == "neg") return -subtree;
		if (head == "sqr") return pow(subtree, 2);
		if (head == "exp") return exp(subtree);
		if (head == "sin") return sin(subtree);
		if (head == "cos") return cos(subtree);
		if (head == "tan") return tan(subtree);
		if (head == "log") return log(subtree);
	}

	// binary operators
	if (rest.size() == 2) {
		// Compute the two subtrees
		double left = evaluate(rest[0], t);
		double right = evaluate(rest[1], t);

		// Special Case
		if (head == "hash") return elegentPair(left, right);

		// Normal Cases
		if (head == "+") return left + right;
		if (head == "-") return left - right;
		if (head == "*") return left * right;
		if (head == "/") return left / right;
		if (head == "max") return max(left, right);
		if (head == "min") return min(left, right);
	}

	// ternary operator
	if (rest.size() == 3) {
		// simulate "if"
		if (evalCond(rest[0], t)) return evaluate(rest[1], t);
		else return evaluate(rest[2], t);
	}
	return 0;
}

extern "C" {
	// double extParseEval(char*& funcStr, double& x, double& y, double& t) {
	double extParseEval(char*& funcStr, double& t) {
		// convert the funcStr to C++'s string
		string funcString = funcStr;
		return evaluate(funcString, t);
	}
}


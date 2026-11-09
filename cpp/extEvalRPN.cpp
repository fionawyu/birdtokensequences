// This is a C++ implementation of the reverse polish notation calculator

// Compile as mex mexEvalFormula.cpp 

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <stack>
#include <cmath>
#include <iomanip>

using namespace std;

typedef int64_t i64;

string doubleToString(const double& value, const int& precision = 17) {
    ostringstream oss;
    oss << fixed << setprecision(precision) << value;
    return oss.str();
}

vector<string> strSplit(const string& formula) {
	vector<string> words;
	istringstream stream(formula);
	string word; // holder variable
	while (stream >> word) words.push_back(word);
	return words;
}

bool is_number(const string& str) {
    try {
        stod(str);  // Try to convert to a double
        return true;     // If successful, it's a number
    } catch (invalid_argument& e) {
        return false;    // If it throws an exception, it's not a number
    } catch (out_of_range& e) {
        return false;    // If the number is too large, return false
    }
}

double maxAll(const vector<double>& vals) {
	double ret = vals[0];
	if (vals.size() == 1) return ret;
	for (i64 i = 1; i < vals.size(); i++) {
		if (vals[i] > ret) ret = vals[i];
	}
	return ret;
}

double minAll(const vector<double>& vals) {
	double ret = vals[0];
	if (vals.size() == 1) return ret;
	for (i64 i = 1; i < vals.size(); i++) {
		if (vals[i] < ret) ret = vals[i];
	}
	return ret;
}

double argmaxAll(const vector<double>& vals) {
	double ret = 1; // start from index 1 instead of 0
	if (vals.size() == 1) return ret;
	double largestVal = vals[0];
	for (i64 i = 1; i < vals.size(); i++) {
		if (vals[i] > largestVal) {
			largestVal = vals[i];
			ret = i;
		}
	}
	return ret;
}

double argminAll(const vector<double>& vals) {
	double ret = 1; // start from index 1 instead of 0
	if (vals.size() == 1) return ret;
	double smallestVal = vals[0];
	for (i64 i = 1; i < vals.size(); i++) {
		if (vals[i] < smallestVal) {
			smallestVal = vals[i];
			ret = i;
		}
	}
	return ret;
}

string stkPop(stack<string>& stk) {
	string topEle;
	if (stk.empty()) return "NaN";
	else {
		topEle = stk.top();
		stk.pop();
	}
	return topEle;
}

void runFormula(const string& formula, const double* input, double& output, double& stackleft) {
	vector<string> elements = strSplit(formula);
	stack<string> stk;
	string a, b, c, d, e;
	double val;
	vector<double> vals;
	for (string op : elements) {
		// Grass
        if (op == "t") {
			stk.push(doubleToString(input[0]));
        } else if (op == "V") {
			stk.push(doubleToString(input[1]));
        } else if (op == "pi") {
			stk.push(doubleToString(3.1415926));
        } else if (op == "x") {
			stk.push(doubleToString(input[0]));
		} else if (op == "a") {
			stk.push(doubleToString(input[0]));
		} else if (op == "b") {
			stk.push(doubleToString(input[1]));
		} else if (op == "c") {
			stk.push(doubleToString(input[2]));
		} else if (op == "d") {
			stk.push(doubleToString(input[3]));
		} else if (op == "g_\\leftarrow") {
			stk.push(doubleToString(input[0]));
		} else if (op == "g_\\uparrow") {
			stk.push(doubleToString(input[1]));
		} else if (op == "g_\\rightarrow") {
			stk.push(doubleToString(input[2]));
		} else if (op == "g_\\downarrow") {
			stk.push(doubleToString(input[3]));
		} else if (op == "g_+") {
			stk.push(doubleToString(input[4]));
		} else if (op == "e") { // energy
			stk.push(doubleToString(input[5]));
		} else if (op == "a") { // age
			stk.push(doubleToString(input[6]));
		} else if (op == "+") {
			b = stkPop(stk);
			a = stkPop(stk);
			if (!is_number(a) || !is_number(b)) stk.push("NaN");
			else stk.push(doubleToString(stod(a) + stod(b)));
		} else if (op == "-") {
			b = stkPop(stk);
			a = stkPop(stk);
			if (!is_number(a) || !is_number(b)) stk.push("NaN");
			else stk.push(doubleToString(stod(a) - stod(b)));
		} else if (op == "*") {
			b = stkPop(stk);
			a = stkPop(stk);
			if (!is_number(a) || !is_number(b)) stk.push("NaN");
			else stk.push(doubleToString(stod(a) * stod(b)));
		} else if (op == "/") {
			b = stkPop(stk);
			a = stkPop(stk);
			val = stod(a) / stod(b);
			if (isinf(val)) stk.push("NaN");
			else stk.push(doubleToString(val));
		} else if (op == "pow" || op == "^") {
			b = stkPop(stk);
			a = stkPop(stk);
			val = pow(stod(a), stod(b));
			if (!isnan(val) && !isinf(val)) stk.push(doubleToString(val));
			else stk.push("NaN");
		} else if (op == "++") {
			a = stkPop(stk);
			stk.push(doubleToString(stod(a) + 1));
		} else if (op == "--") {
			a = stkPop(stk);
			stk.push(doubleToString(stod(a) - 1));
		} else if (op == "sqrt") {
			a = stkPop(stk);
			stk.push(doubleToString(sqrt(stod(a))));
		} else if (op == "sqr") {
			a = stkPop(stk);
			double da = stod(a);
			stk.push(doubleToString(da * da));
		} else if (op == "sin") {
			a = stkPop(stk);
			stk.push(doubleToString(sin(stod(a))));
		} else if (op == "cos") {
			a = stkPop(stk);
			stk.push(doubleToString(cos(stod(a))));
		} else if (op == "tan") {
			a = stkPop(stk);
			stk.push(doubleToString(tan(stod(a))));
		} else if (op == "abs") {
			a = stkPop(stk);
			stk.push(doubleToString(abs(stod(a))));
		} else if (op == "relu") {
			a = stkPop(stk);
			stk.push(doubleToString(max(stod(a), 0.0)));
		} else if (op == "round") {
			a = stkPop(stk);
			stk.push(doubleToString(round(stod(a))));
		} else if (op == "neg") {
			a = stkPop(stk);
			stk.push(doubleToString(-stod(a)));
		} else if (op == "exp") {
			a = stkPop(stk);
			stk.push(doubleToString(exp(stod(a))));
		} else if (op == "log") {
			a = stkPop(stk);
			stk.push(doubleToString(log(stod(a))));
		} else if (op == "max_2") {
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			stk.push(doubleToString(maxAll(vals)));
		} else if (op == "max_3") {
			c = stkPop(stk);
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			if (is_number(c)) vals.push_back(stod(c));
			stk.push(doubleToString(maxAll(vals)));
		} else if (op == "max_4") {
			d = stkPop(stk);
			c = stkPop(stk);
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			if (is_number(c)) vals.push_back(stod(c));
			if (is_number(d)) vals.push_back(stod(d));
			stk.push(doubleToString(maxAll(vals)));
		} else if (op == "max_5") {
			e = stkPop(stk);
			d = stkPop(stk);
			c = stkPop(stk);
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			if (is_number(c)) vals.push_back(stod(c));
			if (is_number(d)) vals.push_back(stod(d));
			if (is_number(e)) vals.push_back(stod(e));
			stk.push(doubleToString(maxAll(vals)));
		} else if (op == "min_2") {
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			stk.push(doubleToString(minAll(vals)));
		} else if (op == "min_3") {
			c = stkPop(stk);
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			if (is_number(c)) vals.push_back(stod(c));
			stk.push(doubleToString(minAll(vals)));
		} else if (op == "min_4") {
			d = stkPop(stk);
			c = stkPop(stk);
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			if (is_number(c)) vals.push_back(stod(c));
			if (is_number(d)) vals.push_back(stod(d));
			stk.push(doubleToString(minAll(vals)));
		} else if (op == "min_5") {
			e = stkPop(stk);
			d = stkPop(stk);
			c = stkPop(stk);
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			if (is_number(c)) vals.push_back(stod(c));
			if (is_number(d)) vals.push_back(stod(d));
			if (is_number(e)) vals.push_back(stod(e));
			stk.push(doubleToString(minAll(vals)));
		} else if (op == "argmax_2") {
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			stk.push(doubleToString(argmaxAll(vals)));
		} else if (op == "argmax_3") {
			c = stkPop(stk);
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			if (is_number(c)) vals.push_back(stod(c));
			stk.push(doubleToString(argmaxAll(vals)));
		} else if (op == "argmax_4") {
			d = stkPop(stk);
			c = stkPop(stk);
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			if (is_number(c)) vals.push_back(stod(c));
			if (is_number(d)) vals.push_back(stod(d));
			stk.push(doubleToString(argmaxAll(vals)));
		} else if (op == "argmax_5") {
			e = stkPop(stk);
			d = stkPop(stk);
			c = stkPop(stk);
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			if (is_number(c)) vals.push_back(stod(c));
			if (is_number(d)) vals.push_back(stod(d));
			if (is_number(e)) vals.push_back(stod(e));
			stk.push(doubleToString(argmaxAll(vals)));
		} else if (op == "argmin_2") {
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			stk.push(doubleToString(argminAll(vals)));
		} else if (op == "argmin_3") {
			c = stkPop(stk);
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			if (is_number(c)) vals.push_back(stod(c));
			stk.push(doubleToString(argminAll(vals)));
		} else if (op == "argmin_4") {
			d = stkPop(stk);
			c = stkPop(stk);
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			if (is_number(c)) vals.push_back(stod(c));
			if (is_number(d)) vals.push_back(stod(d));
			stk.push(doubleToString(argminAll(vals)));
		} else if (op == "argmin_5") {
			e = stkPop(stk);
			d = stkPop(stk);
			c = stkPop(stk);
			b = stkPop(stk);
			a = stkPop(stk);
			if (is_number(a)) vals.push_back(stod(a));
			if (is_number(b)) vals.push_back(stod(b));
			if (is_number(c)) vals.push_back(stod(c));
			if (is_number(d)) vals.push_back(stod(d));
			if (is_number(e)) vals.push_back(stod(e));
			stk.push(doubleToString(argminAll(vals)));
		} else stk.push(op); // op is a constant, convert to double and push
	}
	output = stod(stk.top()); stk.pop();
	stackleft = stk.size();
}

extern "C" {
	void extEvalRPN(char*& funcStr, double** inputs, i64& size, double*& outputs, double*& stacklefts) {
		// convert the funcStr to C++'s string
		string funcString = funcStr;

		// allocate the resulting array
		outputs = (double*)malloc(size * sizeof(double));
		stacklefts = (double*)malloc(size * sizeof(double));

		// evaluate on all inputs
		for (i64 i = 0; i < size; i++) {
			double output = 0;
			double stackleft = 0;
			runFormula(funcString, inputs[i], output, stackleft);
			outputs[i] = output;
			stacklefts[i] = stackleft;
		}
	}
}




#include <iostream>
#include <sstream>
#include <vector>
#include <map>

using namespace std;

map<string, int> angka = {
    {"nol", 0}, {"satu", 1}, {"dua", 2},
    {"tiga", 3}, {"empat", 4}, {"lima", 5},
    {"enam", 6}, {"tujuh", 7}, {"delapan", 8},
    {"sembilan", 9}
};

int parseNumber(string input) {
    stringstream ss(input);
    vector<string> words;
    string word;

    while (ss >> word)
        words.push_back(word);

    // "dua nol tiga" -> 203
    bool digitMode = true;

    for (auto &w : words) {
        if (!angka.count(w)) {
            digitMode = false;
            break;
        }
    }

    if (digitMode) {
        int result = 0;

        for (auto &w : words)
            result = result * 10 + angka[w];

        return result;
    }

    // "dua ratus tiga" -> 203
    int result = 0;
    int current = 0;

    for (auto &w : words) {
        if (angka.count(w)) {
            current = angka[w];
        }
        else if (w == "ratus") {
            result += current * 100;
            current = 0;
        }
    }

    return result + current;
}

int main() {
    cout << parseNumber("dua ratus tiga") << endl;
    cout << parseNumber("dua nol tiga") << endl;
}
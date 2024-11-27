#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

std::vector<int> read_from_file(std::string path) {
    std::fstream file(path, std::ios::in | std::ios::binary);
    std::vector<int> vec;

    if (!file) {
        std::cout << "Input File missing" << std::endl;
        std::exit(1);
    }

    int x;
    while (file.read((char *) &x, sizeof(x))) {
        vec.push_back(x);
    }

    file.close();
    return vec;
}

void write_to_file(std::string path, std::vector<int> &vec) {
    std::fstream file(path, std::ios::out | std::ios::binary);

    if (!file) {
        std::cout << "Output File missing" << std::endl;
        std::exit(1);
    }

    for (int i = 0; i < vec.size(); i += 1) {
        file.write((char *) &vec[i], sizeof(vec[i]));
    }

    file.close();
}

#ifndef WRITER_H
#define WRITER_H

#include <stdio.h>
#include <tree_sitter/api.h>

void write_tree(FILE *, const char *, TSNode);
void write_children(FILE *, const char *, TSNode);

#endif

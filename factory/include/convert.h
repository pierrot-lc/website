#ifndef CONVERT_H
#define CONVERT_H

#include <stdio.h>
#include <tree_sitter/api.h>

/* Main function, used to convert the content of the markdown tree into the
 * given file.
 */
void convert_tree(FILE *, const char *, TSNode);

void convert_heading(FILE *, const char *, TSNode);
void convert_inline(FILE *, const char *, TSNode);
void convert_named_children(FILE *, const char *, TSNode);

#endif

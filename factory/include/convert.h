#ifndef CONVERT_H
#define CONVERT_H

#include <tree_sitter/api.h>

#include "tree.h"

/* Main function, used to convert the content of the markdown tree into the
 * given file.
 */
Node *convert_tree_md(const char *, TSTree *);

/* Inline are their own tree.
 */
Node *convert_tree_md_inline(const char *, TSTree *);

Node *_heading(const char *, TSNode);
Node *_inline(const char *, TSNode);
Node *_node_md(const char *, TSNode);
Node *_node_md_inline(const char *, TSNode);
void _children(Node *, const char *, TSNode);
// void _convert_emphasis(FILE *file, const char *source, TSNode node);
// void _convert_inline(FILE *, const char *, TSNode);
// void _convert_named_children(FILE *, const char *, TSNode);

#endif

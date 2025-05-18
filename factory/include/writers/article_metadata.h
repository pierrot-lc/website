#ifndef WRITERS_ARTICLE_MEATADATA
#define WRITERS_ARTICLE_MEATADATA

#include <stdio.h>

#include "tree.h"

/**
 * Write to the file the article's metadata found in the tree.
 */
void write_article_metadata(FILE *file, Node *tree);

#endif

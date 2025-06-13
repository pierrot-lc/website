#ifndef WRITERS_BODY_H
#define WRITERS_BODY_H

#include <stdio.h>

#include "tree.h"

/**
 * Write the tree as its HTML representation.
 */
void write_tree(FILE *file, Node *tree);

/**
 * Write the general page information such as its title and its tags.
 *
 * That information is fetch from the YAML nodes in the tree.
 */
void write_page_info(FILE *file, Node *tree);

/**
 * Write the header section of the page.
 */
void write_header(FILE *file, Node *tree);

#endif

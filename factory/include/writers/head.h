#ifndef WRITERS_HEAD_H
#define WRITERS_HEAD_H

#include <stdio.h>

#include "tree.h"

/**
 * Write the head section based on the YAML nodes.
 */
void write_head(FILE *file, Node *tree);

#endif

#ifndef WRITERS_HTML_H
#define WRITERS_HTML_H

#include <stdio.h>

#include "tree.h"

/* Write the whole HTML page. */
void write_html(FILE *file, Node *node);

#endif

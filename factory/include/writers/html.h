#ifndef WRITERS_HTML_H
#define WRITERS_HTML_H

#include <stdio.h>

#include "tree.h"

void articles_index_page(FILE *file, const char *articles_dir, Node *config);

/* Write the whole HTML page. */
void write_html(FILE *file, Node *node);

#endif

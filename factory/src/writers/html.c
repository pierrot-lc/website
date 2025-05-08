#include <stdio.h>

#include "tree.h"
#include "writers/body_main.h"
#include "writers/head.h"

void write_html(FILE *file, Node *node) {
  fprintf(file, "<html>\n\n");

  write_head(file, node);

  fprintf(file, "\n<body>\n");
  write_body_main(file, node);
  fprintf(file, "</body>\n\n");

  fprintf(file, "</html>\n");
}

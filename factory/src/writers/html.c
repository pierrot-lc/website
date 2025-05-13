#include <stdio.h>

#include "tree.h"
#include "writers/body_main.h"
#include "writers/head.h"
#include "writers/header.h"

void write_html(FILE *file, Node *tree) {
  fprintf(file, "<!DOCTYPE html>\n");
  fprintf(file, "<html>\n\n");

  write_head(file, tree);

  fprintf(file, "\n<body>\n");
  write_header(file, tree);
  write_body_main(file, tree);
  fprintf(file, "</body>\n\n");

  fprintf(file, "</html>\n");
}

// Wait for the page to be loaded and then parse all latex spans to render
// their LaTeX expressions using KaTeX.

document.addEventListener("DOMContentLoaded", function () {
  const mathElements = document.querySelectorAll(".latex-inline");

  mathElements.forEach(function (element) {
    const latex = element.textContent;
    katex.render(latex, element, {
      displayMode: false,
    });
  });
});

document.addEventListener("DOMContentLoaded", function () {
  const mathElements = document.querySelectorAll(".latex-display");

  mathElements.forEach(function (element) {
    const latex = element.textContent;
    katex.render(latex, element, {
      displayMode: true,
    });
  });
});

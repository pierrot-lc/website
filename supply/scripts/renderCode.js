// Replace lines marked by a comment 'mark-line' with encapsulated <mark>line</mark>.
// From https://github.com/highlightjs/highlight.js/issues/740#issuecomment-1298487876
hljs.addPlugin({
    'after:highlightElement': ({ el, result, text }) => {
        let html = el.innerHTML;
        let markedHtml = html.replace(/^(\s*)(.+?)\s*<span class="hljs-comment">.*?\bmark-line\b.*?<\/span>$/mg, '$1<mark>$2</mark>');

        if (html != markedHtml) {
            el.innerHTML = markedHtml;
        }
    }
});

hljs.highlightAll();

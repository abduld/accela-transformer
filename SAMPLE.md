# Implementing a Transformer Block using Accera

Docsxxify :fas fa-home fa-fw:
 uses [Prism](https://prismjs.com) to highlight code blocks in your pages. Prism supports the following languages by default:

* Markup - `markup`, `html`, `xml`, `svg`, `mathml`, `ssml`, `atom`, `rss`
* CSS - `css`
* C-like - `clike`
* JavaScript - `javascript`, `js`


```ditaa 
      +--------+
      |        |
      |  User  |
      |        |
      +--------+
          ^
  request |
          v
  +-------------+
  |             |
  |    Kroki    |
  |             |---+
  +-------------+   |
       ^  ^         | inflate
       |  |         |
       v  +---------+
  +-------------+
  |             |
  |    Ditaa    |
  |             |----+
  +-------------+    |
             ^       | process
             |       |
             +-------+
```




Support for [additional languages](https://prismjs.com/#supported-languages) is available by loading the language-specific [grammar files](https://cdn.jsdelivr.net/npm/prismjs@1/components/) via CDN:

```html
<script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-bash.min.js"></script>
<script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-php.min.js"></script>
```

To enable syntax highlighting, wrap each code block in triple backticks with the [language](https://prismjs.com/#supported-languages) specified on the first line:

````
```html
<p>This is a paragraph</p>
<a href="//docsify.js.org/">Docsify</a>
```

```bash
echo "hello"
```

```php
function getAdder(int $x): int 
{
    return 123;
}
```
````

The above markdown will be rendered as:

```html
<p>This is a paragraph</p>
<a href="//docsify.js.org/">Docsify</a>
```

```bash
echo "hello"
```

```php
function getAdder(int $x): int 
{
    return 123;
}
```
 


## Demo

<!-- div:left-panel -->

If you are on widescreen, checkout the *right* panel, *right* there →

<!-- div:right-panel -->

This is an example panel.

You can see it's usage in practice in the docs listed below:

-   [Fairlay API](https://fairlay.com/api)
-   [FLAP services](https://docs.flap.cloud/#/create_new_service?id=special-files)

<small>please contact me if you use docsify-example-panels. i would like to display it here too.</small>

<!-- div:title-panel -->

## Features

<!-- div:left-panel -->

**Advantages**

-   Create div panels really fast and anywhere in your .md file.
-   Choose the classnames for your divs and stylize them.
-   Use CSS custom properties to change it's structure.
-   Prefab CSS classes for "left-panel", "right-panel" and "title-panel".

**Compatibility**

-   Fully compatible with any markdown or html features:

<details>
  <summary>code snippets </summary>

```html
  <body>
    <img src="http://www.pudim.com.br/pudim.jpg">
  </body>
```

</details>

<details>
  <summary>quotes</summary>

> just a quote

?> a cooler quote...  <small> (at least i think it is)</small>

</details>

<details>
  <summary>images <small>(memorable)</small></summary>

  <br/>
  <img src="https://avatars0.githubusercontent.com/u/5666881?s=400&u=d94729bdf16611396a720b338c115ec0be656ba6&v=4" width="64" height="64">
</details>

-   Fully compatible with major docsify plugins such as:

> [docsify-themeable](https://jhildenbiddle.github.io/docsify-themeable/)
>
> [docsify-tabs](https://jhildenbiddle.github.io/docsify-tabs/)
>
> [docsify-copy-code](https://github.com/jperasmus/docsify-copy-code)
>
> [docsify-pagination](https://github.com/imyelo/docsify-pagination)
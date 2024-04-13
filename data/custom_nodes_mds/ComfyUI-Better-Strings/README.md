# ComfyUI Better Strings

Strings should be easy, and simple.

This repository aims to provide a set of nodes that make working with strings in ComfyUI a little bit easier.

As it currently stands, it adds a single, extremely basic node: The Better String.

In the future, I hope to add support for better string concatenation as well as simple A1111 style wildcards.

## Better String

The Better String node is a simple node that gives you access to a primitive multi-line string.

_"But there's already a primitive string node, why do I need this?"_

Well, you don't need this. But you might want this if you are sick and tired of your primitive node defaulting to a singleline string and cutting off 
half of the content that you've entered.

Example:

> Convert the filename of the Save Image node to a widget and drag in a primitive string node.
>
> See that, once the string gets sufficiently large, it cuts off half of the string. This is most frustrating when using variables to name files,
> such
> as your checkpoint name, datetime, or prompt details.
>
> Now, you can plug in a Better String node and it will automatically default to a multiline string, allowing you to see the entire string.

It also allows you to connect the primitive to rerouting nodes way easier, if you're into that sort of thing.

![image](https://github.com/HaydenReeve/ComfyUI-Better-Strings/assets/39004735/5c04cb66-ab32-44ea-b1e8-67b0b674cca9)

# Installation

Navigate to ComfyUI/custom_nodes in your terminal of choice, then run the following command:

```bash
git clone https://github.com/HaydenReeve/ComfyUI-Better-Strings/
```

That's it!

You'll find my nodes under the heading `Better Things 💡`

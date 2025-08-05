# Instructions for the Configuration System

The configuration system I used here is based on [Hydra](https://hydra.cc/). However, I made some small 'hack' to achieve some better features, such as `_hub_`. It might be a little bit hard to understand at first, but a comprehensive guidance is provided. Check `README.md` in each directory for more details.

## Philosophy

- Easy to modify and maintain.
- Help you to code with clear structure.
- Consistency.
- Easy to trace and identify specific item.

## Some Ideas

- Less ListConfig, or ListConfig only for real list data.
    - Dumped list will be unfolded, each element occupies one line, and it's annoying when presenting.
    - List things are not friendly to command line arguments supports.
- For defaults list, `_self_` must be explicitly specified.
    - Items before `_self_` means 'based on those items'.
    - Items after `_self_` means 'import those items'.

### COMPOSITION OVER INHERITANCE

Do not use "overrides" AS MUCH AS POSSIBLE, except when the changes are really tiny. Since it's hard to identify which term is actually used without running the code. Instead, the `default.yaml` serves like a template, you are supposed to copy it and modify it to create a new configuration.

### REFERENCE OVER COPY

If you want to use one things for many times (across each experiments or across each components in one experiment), you'd better use `${...}` to reference the Hydra object. So that you only need to modify one place while ensuring the consistency. Think about where to put the source of the object, `_hub_` is recommended but may not be the best choice every time.

### PREPARE EVERYTHING YOU NEED LOCALLY

Sometimes you might want to use some configurations outside the class configuration (I mean in the coding process). In that case, I recommend you to reference these things again in local configuration package. It might be too redundant, but it will bring cleaner code.
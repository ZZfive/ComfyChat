# ClipStuff
## Basic Instructions.
Clone repo into custom_nodes folder.
Install the requirements.txt file via pip.

Pass CLIP output from Load Checkpoint into SpecialClipLoader node, then use the outputted clip with standard Clip Text Encode.

See example workflow in examples folder.

### Example Photo
![Example Photo](assets/first_example.png)

## Functions

## Syntax Elements

1. **Embedding**:
    - Syntax: `embedding:WORD`
    - Example: `embedding:face_vector`
    - Represents a named vector embedding(Textual Inversion).

2. **Word**:
    - Syntax: Any alphanumeric word including characters such as `,`, `_`, and `-`.
    - Example: `cat, dog_face, id_123`
    - Represents simple words or identifiers.

3. **Quoted String**:
    - Syntax: A string enclosed within double or single quotes. You can escape quotes inside the string using a backslash (`\`).
    - Example: `"Hello World"`, `'It\'s a sunny day'`
    - Represents string literals.


### Notes on Arguments:
- Each function takes one or more arguments.
- An argument (`arg`) can be an embedding, multiple words, another function, or a quoted string.
- For functions that accept multiple arguments, they are separated by the `|` symbol.

## Examples

1. Add two embeddings and normalize the result:
   ```
   norm(sum(cat | dog | horse | parrot))
   ```

2. Negate an embedding:
   ```
   neg(embedding:body_vector)
   ```

3. King - Man + Woman = Queen:
   ```
   sum(diff(king|man)|woman)
   ```
   or
   ```
   sum(king|neg(man)|woman)
   ```

## Functions

Here are the available functions and their usage:

| Display Name | Action Name | Description | Usage Examples |
| --- | --- | --- | --- |
| Multiply | mult | Multiplies the provided segments or actions by the multiplier. | <ul><li>mult(The cat is\|2.5)</li><li>mult(Cat\|-1)</li></ul> |
| Set Dimensions | setDims | Sets the specified dimensions of the input embeddings to the specified value | <ul><li>The setDims(cat\|4, -0.01253\|76, 1.2) is happy</li></ul> |
| Negate | neg | Negates the provided segments or actions. | <ul><li>neg(cat)</li><li>sum(king\|neg(man)\|women)</li></ul> |
| Normalize | norm | Normalizes the provided segments or actions. | <ul><li>norm(cat)</li><li>sum(cat\|norm(sum(tiger\|fish)))</li></ul> |
| Positional Embedding Scale | posScale | Scales(Multiplies) the positional embeddings of the provided segments or actions by the multiplier. | <ul><li>A posScale(cat\|1.5) on a rainy day</li></ul> |
| Random Embedding | rand | Returns a random embedding of the specified token length, with the values optionally bounded by the second and third arguments. | <ul><li>A rand(1) cat</li><li>A rand(1\|-1\|1) cat</li></ul> |
| Difference | diff | Subtracts the segments in the order they are given. The first segment is subtracted from the second, then the third from the result, and so on. | <ul><li>diff(The cat is\|The dog is)</li><li>diff(Cat\|Dog)</li><li>sum(diff(king\|man)\|woman)</li></ul> |
| Slerp | slerp | Performs a slerp(Interpolation) between two segments or actions, with the given weight. The recommended weight is 0 - 1 | <ul><li>The slerp(cat\|dog\|0.5) is happy</li></ul> |
| Pooled Average(Experimental) | _exp-pooledAvg | Processes the provided segments or actions fully through CLIP and creates a pooled average of the last hidden state by averaging the last hidden state of each token. | <ul><li>A cat on a _exp-pooledAvg(beautiful sunny day)</li><li>A _exp-pooledAvg(broken glass) bottle</li></ul> |
| Sum | sum | Adds the embeddings of the provided segments or actions. | <ul><li>A happy sum(cat\|dog\|shark)</li></ul> |
| Pooler Output(Experimental) | _exp-pooler | Processes the provided segments or actions fully through CLIP and returns the pooler_output from the transformer | <ul><li>A cat on a _exp-pooler(beautiful sunny day)</li><li>A _exp-pooler(broken glass) bottle</li></ul> |
| Scale Dimensions | scaleDims | Scales the specified dimensions of the input embeddings by the specified amount | <ul><li>The scaleDims(cat\|4,1.5\|76,1.2) is happy</li></ul> |
| Ignore Positional Embeddings | postPos | Prevents positional embeddings from being applied to the provided segments or actions. | <ul><li>A postPos(cat) on a rainy day</li></ul> |
| Average | avg | Performs a weighted average between two segments or actions. The recommended weight is 0 - 1. | <ul><li>avg(The cat is\|The dog is\|0.5)</li><li>avg(Cat\|Dog\|0.5)</li></ul> |


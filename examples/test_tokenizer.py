import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import tensor_engine as te
logger.info('Has Tokenizer? %s', hasattr(te,'Tokenizer'))
import os
tokenizer_path = r'E:\Tensor-Engine\examples\Llama-3.2-1B\tokenizer.json'
if not os.path.exists(tokenizer_path):
    logger.info('Tokenizer file %s not found; skipping tokenizer example', tokenizer_path)
else:
    t = te.Tokenizer.from_file(tokenizer_path)
    logger.info('vocab size %d', t.vocab_size())
if 't' in locals():
    ids = t.encode('<|begin_of_text|> Hello')
    logger.info('ids %s', ids[:10])
    logger.info('decode %s', t.decode(ids[:10]))
else:
    logger.info('Tokenizer example skipped (no tokenizer loaded)')

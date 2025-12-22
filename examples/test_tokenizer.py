import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import tensor_engine as te
logger.info('Has Tokenizer? %s', hasattr(te,'Tokenizer'))
t = te.Tokenizer.from_file(r'E:\Tensor-Engine\examples\Llama-3.2-1B\tokenizer.json')
logger.info('vocab size %d', t.vocab_size())
ids = t.encode('<|begin_of_text|> Hello')
logger.info('ids %s', ids[:10])
logger.info('decode %s', t.decode(ids[:10]))

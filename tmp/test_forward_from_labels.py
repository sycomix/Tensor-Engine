import tensor_engine as te
Tensor = getattr(te, 'Tensor', None)
Labels = getattr(te, 'Labels', None)
SoftmaxCrossEntropyLoss = getattr(te, 'SoftmaxCrossEntropyLoss', None)
logits = None
if Tensor is not None:
    logits = Tensor([1.0,2.0,-1.0],[1,3])

if Tensor is not None and Labels is not None and SoftmaxCrossEntropyLoss is not None:
    # Create labels instance in a robust way; Labels may be a constructor or a factory
    labels = None
    # If Labels is directly callable (like a class or function), try to call it
    if callable(Labels):
        try:
            labels = Labels([0,2])
        except Exception:
            labels = None
    # If the direct call failed or Labels isn't callable, try known factory methods
    if labels is None:
        for fname in ('from_list', 'from_array', 'tensor', 'from_iterable', 'from_sequence'):
            if hasattr(Labels, fname) and callable(getattr(Labels, fname)):
                labels = getattr(Labels, fname)([0,2])
                break
    if labels is None:
        print('skip')
    else:
        loss5 = SoftmaxCrossEntropyLoss().forward_from_labels(logits, labels)
    try:
        loss5.backward()
        print('OK', loss5.get_data())
    except Exception as e:
        print('Exception', e)
else:
    print('skip')

# skelify-spin

Define the SKELifySPIN callback process inherited from /pipeline/skelify-refiner@skelify (inheritance + definition mode supported by Hydra)

i80: The interval amplitude is 80 (significantly reducing the number of pseudo-tags generated and minimizing the noise interference of pseudo-tags)
i10kb1: The interval range is 10. A new parameter, max_batches_per_round, has been added. Only the latest pseudo-label is updated each time (saving time).
i230kb1: Interval range 230+ no backtracking (significantly saves computing power)
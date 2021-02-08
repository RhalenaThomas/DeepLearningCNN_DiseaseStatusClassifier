from test_model import test

def main():
    # list of models to run
    model_li = ["batch_1_4", "batch_2_3a_SECOND_TRY", "batch_3_a", "batch_3_a_b", "batch_3_b", "batch_3a_4", "batch_4_plate_1",  "batchfive", "merged_batch_1_batch_2", "model_batch_1_3a", "model_batch_1_3b", "model_XCL_NPC_1", "model_XCL_NPC_2"]
    # list of test sets to run
    test_set_li = ["batch_1_3a", "batch_1_3b", "batch_1_4", "batch_2_3a", "batch_2_3b", "batch_3_a", "batch_3_b", "batch_3_a_b", "batch_3a_4", "batch_3b_4", "batch_4_plate_1", "batchfive", "XCL_NPC_batch_1", "XCL_NPC_batch_2", "batch1and2"]

    file = "results.txt" # text file to append results to

    for model in model_li:
        for test_set in test_set_li:
            acc, report = test(test_set, model)
            f = open(file, "a")
            f.write("the accuracy for model " + model + " and test set " + test_set + " is " + str(acc) + "\n")
            f.write(report + "\n")
            f.write("..............................................................................\n")

            f.close()

if __name__ == '__main__':
   main()
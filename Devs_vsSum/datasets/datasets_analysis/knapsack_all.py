

def print_solutions(current_item, knapsack, current_sum, solution, items, limit):
    #if all items have been processed print the solution and return:
    if current_item == len(items):
        # print knapsack
        # if len(knapsack) > 3:
        solution.append(knapsack)
        return
        # return solution

    #don't take the current item and go check others
    print_solutions(current_item + 1, list(knapsack), current_sum, solution, items, limit)

    #take the current item if the value doesn't exceed the limit
    if (current_sum + items[current_item] <= limit):
        knapsack.append(current_item)
        current_sum += items[current_item]
        #current item taken go check others
        print_solutions(current_item + 1, knapsack, current_sum, solution, items, limit)
if __name__ == '__main__':

    items = [1,1,3,4,5]
    knapsack = []
    limit = 7
    solution=[]
    print_solutions(0,knapsack,0, solution, items, limit)
    print 'DE'
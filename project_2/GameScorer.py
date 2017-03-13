import GameManager
import math

def main():

    # run game ----------------------------------
    games_tot = 25
    results = []
    temp_sum = 0
    temp_len = 0
    for g in range(games_tot):
        temp = GameManager.main()
        results.append(temp[0])
        print ("".ljust(20,"-"))

    for g in range(games_tot):
        print ("game %2.0f, result %s" % (g, results[g]))
        if type(results[g]) == int:
            temp_sum = temp_sum + results[g]
            temp_len +=  1

    if temp_len > 0:
        my_average_result = temp_sum / temp_len
        print ("".ljust(20,"-"))
        print ("my average result %4.0f" % my_average_result)

        # calc grade ----------------------------------
        random_result = 110
        random_grade = 0
        top_result = 1024 + (2048-1024)/2
        top_grade = 200
        if my_average_result <= random_result:
            my_grade = random_grade
        elif my_average_result >= top_result:
            my_grade = top_grade
        else:
            # grade = a * log(result) + b
            a = top_grade / (math.log10(top_result) - math.log10(random_result))
            b = - a * math.log10(random_result)
            my_grade = a * math.log10(my_average_result) + b
        print ("my grade estimation %4.0f / 200" % my_grade)

main()
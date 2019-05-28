file_in = "/space/andyshih2/RF_NB_SDD/output/week9/week9_2.sdd"
file_out = "/space/andyshih2/RF_NB_SDD/output/week9/week9.sdd"

reorder_map = [('baseline_age_0', 31), ('baseline_age_1', 32), ('bmi_BL_0', 14), ('sum_tfeq_f1_0', 29), ('sum_tfeq_f1_1', 30), ('custom_item_0', 24), ('custom_item_1', 25), ('custom_item_2', 26), ('custom_item_3', 27), ('custom_item_4', 28), ('recipe_item_0', 20), ('recipe_item_1', 21), ('recipe_item_2', 22), ('num_of_items_0', 11), ('num_of_items_1', 12), ('num_of_items_2', 13), ('sat_fat_0', 3), ('sat_fat_1', 4), ('sat_fat_2', 5), ('sat_fat_3', 6), ('fv_points_0', 7), ('fv_points_1', 8), ('fv_points_2', 9), ('fv_points_3', 10), ('delta_fat_0', 36), ('delta_fat_1', 37), ('delta_fat_2', 38), ('delta_fat_3', 39), ('delta_fat_4', 40), ('delta_fat_5', 41), ('delta_fat_6', 42), ('delta_fat_7', 43), ('delta_fat_8', 44), ('min_0', 1), ('min_1', 2), ('activity_items_num_0', 18), ('activity_items_num_1', 19), ('num_days_activity_0', 15), ('num_days_activity_1', 16), ('num_days_activity_2', 17), ('weight_0', 0), ('weight_items_num_0', 23), ('num_days_weight_0', 33), ('num_days_weight_1', 34), ('num_days_weight_2', 35)]
reorder_map = [v for k,v in reorder_map]


with open(file_in, 'r') as f:
  lines = f.readlines()

to_write = []
for l in lines:
  arr = l.strip().split(' ')
  if arr[0] == 'L':
    val = int(arr[-1])
    sgn = 1 if val > 0 else -1
    arr[-1] = str(sgn*(reorder_map.index(abs(val)-1)+1))
  to_write.append(' '.join(arr) + '\n')

with open(file_out, 'w') as f:
  f.write(''.join(to_write))



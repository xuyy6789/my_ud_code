from asyncio.base_futures import _FINISHED
import torch
import numpy as np
import xlwt

result = torch.load('./my_logits.pt')
# print(result)

# self.get_label_words_logits = self.get_label_words_logits_singletoken
# self.__get_prompt_tensor_and_mask = self.__get_prompt_tensor_and_mask_singletoken
# self.__get_prompt_tensor_and_mask(prompt_label_idx)

print(len(result))
print(len(result[0]))
print(result[0][0])

for id, result_id in enumerate(result):

    book = xlwt.Workbook()
    sheet = book.add_sheet(u'sheet1',cell_overwrite_ok=True)
    sheet.write(0, 0, 'id')
    sheet.write(0, 1, 'result')

    for i, num in enumerate(result_id):
        num = num.item()
        sheet.write(i + 1, 0, i + 1)
        sheet.write(i + 1, 1, num)

    book.save('./result/result%d.xls' % (id + 1))

    print("save {%d} finished" % (id + 1))
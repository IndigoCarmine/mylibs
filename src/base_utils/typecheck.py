import functools
import inspect


# 引数・戻り値の型チェックを行うデコレータ。
def type_check(func):
    @functools.wraps(func)
    def type_check_wrapper(*args, **kwargs):

        func_signature = inspect.signature(func)

        # 引数の型チェック
        args_dict = func_signature.bind(*args, **kwargs)
        for arg_key, arg_value in args_dict.arguments.items():
            # 引数のアノテーションを取得
            arg_annotation = func_signature.parameters[arg_key].annotation
            specified_arg_type = (
                arg_annotation if type(
                    arg_annotation) is type else inspect._empty
            )
            # 引数とアノテーションの型を比較
            if (
                specified_arg_type is not inspect._empty
                and type(arg_value) is not specified_arg_type
            ):
                error_msg = '引数"{}"の型が対応していません。（アノテーション：{}、引数の型：{})'
                raise TypeError(
                    error_msg.format(
                        arg_key, specified_arg_type, type(arg_value))
                )

        # 関数の実行
        results = func(*args, **kwargs)

        # 戻り値の型チェック
        return_annotation = func_signature.return_annotation
        specified_return_type = (
            return_annotation if type(
                return_annotation) is type else inspect._empty
        )
        # 戻り値とアノテーションの型を比較
        if (
            specified_return_type is not inspect._empty
            and type(results) is not specified_return_type
        ):
            error_msg = (
                "戻り値の型が対応していません。（アノテーション：{}、戻り値の型：{})"
            )
            raise TypeError(error_msg.format(
                specified_return_type, type(results)))

        return results

    return type_check_wrapper

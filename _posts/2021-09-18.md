---
layout: post
title:  "printf 구현체의 분석"
date:   2021-09-18 16:39:06
author: djm03178
tags: C,printf
---

## 개요 ##
C언어를 처음 배울 때, 가장 처음으로 구현하게 되는 Hello World 코드에 반드시 들어가는 함수가 있습니다. 바로 `printf` 함수입니다. 처음부터 배울 정도로 기초적인 함수이지만, 다양한 형식의 출력을 담당한다는 점에서 일반적인 프로그램들에서도 필수적으로 사용되는 중요한 함수라고 할 수 있습니다. 게다가 이 함수의 사용법을 정확히 숙지하는 것은 매우 어렵습니다. 출력의 목적지와 문자열의 종류 등에 따라 `printf`, `sprintf`, `vprintf`, `wprintf` 등의 다양한 변형이 있을 뿐 아니라, '%' 문자를 통해 제공되는 출력 서식 또한 매우 다양한 종류와 구체적인 명세가 있어 이 모두를 익히는 것은 어려운 일입니다.

이 글에서는 `printf`가 이 복잡한 서식 문자들을 처리하는 방법을, glibc의 `printf` 구현체를 직접 따라가며 분석해보도록 하겠습니다.

## vfprintf ##
glibc에서 모든 종류의 `printf`들은 서식문자를 처리하기 위해 내부적으로 `vfprintf`라는 함수로 통하게 됩니다. 이 함수는 [vfprintf_internal.c](https://github.com/bminor/glibc/blob/master/stdio-common/vfprintf-internal.c) 파일에 있습니다. 예를 들어, `sprintf`는 다음과 같은 코드를 통해 이 함수를 호출합니다.

```c
int
__vsprintf_internal (char *string, size_t maxlen,
		     const char *format, va_list args,
		     unsigned int mode_flags)
{
  _IO_strfile sf;
  int ret;

  ...

  _IO_str_init_static_internal (&sf, string,
				(maxlen == -1) ? -1 : maxlen - 1,
				string);
  ret = __vfprintf_internal (&sf._sbf._f, format, args, mode_flags);

  *sf._sbf._f._IO_write_ptr = '\0';
  return ret;
}
```

`vfprintf` 함수는 파일을 취급하는 함수이기 때문에 그를 위한 파일 구조체를 만들어주고, 결과를 저장하기 위한 `string` 문자열을 해당 파일의 쓰기 목적지로 설정해준 후 `__vfprintf_internal` 함수를 호출하는 것을 볼 수 있습니다.

이렇게 호출되는 `__vfprintf_internal`, 또는 그냥 `vfprintf` 함수는 다음과 같이 생겼습니다.

```c
# define vfprintf	__vfprintf_internal

...

/* The function itself.  */
int
vfprintf (FILE *s, const CHAR_T *format, va_list ap, unsigned int mode_flags)
{
  ...
}
```

'...' 안에 숨겨진 약 2천 줄에 달하는 코드 및 매크로들이 지금부터 살펴볼 내용입니다.

## 전체적인 구조 ##
`vfprintf` 함수의 파라미터들은 다음과 같습니다.

* `s`: 출력할 파일 스트림입니다.
* `format`: 서식 문자가 담긴 문자열입니다.
* `ap`: 서식 문자에 순서대로 대응되는, 추가로 전해진 인자들입니다.
* `mode_flags`: 특수한 경우에 사용되는 모드들에 대한 플래그로, 일반적으로 라이브러리를 통해 호출하는 경우에는 0입니다.

`vfprintf` 함수의 주 역할은 `format` 문자열에 담긴 서식 문자 각각에 대응하는 인자들을 `ap`로부터 순서대로 가져와 요청된 서식에 맞추어 문자열로 변환하여 `s`에 출력하는 것입니다.

서식 문자들을 찾기 위해 `vfprintf`는 포인터를 사용하여 처음부터 순서대로 서식 문자에 해당하는 값들을 찾아내고, 모든 서식 문자를 처리할 때까지 루프를 돌게 됩니다.

```c
/* Current character in format string.  */
const UCHAR_T *f;

/* End of leading constant string.  */
const UCHAR_T *lead_str_end;

/* Points to next format specifier.  */
const UCHAR_T *end_of_spec;
...

/* Find the first format specifier.  */
f = lead_str_end = __find_specmb ((const UCHAR_T *) format);

...

/* Write the literal text before the first format.  */
outstring ((const UCHAR_T *) format,
     lead_str_end - (const UCHAR_T *) format);

/* If we only have to print a simple string, return now.  */
if (*f == L_('\0'))
  goto all_done;

...

/* Process whole format string.  */
do
  {
    STEP0_3_TABLE;
    STEP4_TABLE;

    ...
    (각 서식 문자에 대한 처리 및 출력)
    ...

    /* Look for next format specifier.  */
    f = __find_specmb ((end_of_spec = ++f));

    /* Write the following constant string.  */
    outstring (end_of_spec, f - end_of_spec);
  }
while (*f != L_('\0'));
```

`__find_specmb` 함수는 현재 위치 이후에서 '%' 문자를 찾아주는 역할을 합니다. 이를 통해 찾아낸 '%'의 위치가 포인터 `f`에 담기고, 그 위치의 서식 문자에 대한 처리와 그에 대한 출력을 do ~ while 루프 내에서 처리하는 것을 반복하게 됩니다. 한 서식 문자에 대한 처리가 끝나면 `outstring` 함수를 사용하여 그 다음 서식 문자까지의 일반 문자열들을 출력합니다.

이 함수의 핵심은 `STEP0_3_TABLE`과 `STEP4_TABLE`과 각 서식 문자를 처리하는 부분들입니다. 이 부분이 어떻게 동작하는지 지금부터 살펴보도록 하겠습니다.

## 점프 테이블 ##
저수준의 C 코드들에서 자주 보이는 것 중 하나는 일반적으로 많이 쓰지 말라고 배우는 goto문입니다. goto문을 남용하면 코드가 스파게티가 되어 실행 흐름을 파악하기 어렵게 되므로 상용 프로그램을 구현할 때에는 그다지 권장되지 않는 것이 사실이지만, 이렇게 밑바닥에서 성능을 최우선시해야 하는 코드들에는 매우 자주 사용되곤 합니다.

`vfprintf` 함수 역시도 다수의 레이블과 goto를 활용하고 있습니다. 윗문단에서 잠깐 나온 `STEP0_3_TABLE`과 `STEP4_TABLE`이 그것인데, 이것들이 어떻게 생겼는지 찾아가 보겠습니다.

```c
#define STEP0_3_TABLE							      \
    /* Step 0: at the beginning.  */					      \
    static JUMP_TABLE_TYPE step0_jumps[30] =				      \
    {									      \
      REF (form_unknown),						      \
      REF (flag_space),		/* for ' ' */				      \
      REF (flag_plus),		/* for '+' */				      \
      REF (flag_minus),		/* for '-' */				      \
      ...
    };
    /* Step 1: after processing width.  */				      \
    static JUMP_TABLE_TYPE step1_jumps[30] =				      \
    {									      \
      REF (form_unknown),						      \
      REF (form_unknown),	/* for ' ' */				      \
      REF (form_unknown),	/* for '+' */				      \
      REF (form_unknown),	/* for '-' */				      \
      ...
    };
    /* Step 2: after processing precision.  */				      \
    static JUMP_TABLE_TYPE step2_jumps[30] =				      \
    {									      \
      REF (form_unknown),						      \
      REF (form_unknown),	/* for ' ' */				      \
      REF (form_unknown),	/* for '+' */				      \
      REF (form_unknown),	/* for '-' */				      \
      ...
    };
    /* Step 3a: after processing first 'h' modifier.  */		      \
    static JUMP_TABLE_TYPE step3a_jumps[30] =				      \
    {									      \
      REF (form_unknown),						      \
      REF (form_unknown),	/* for ' ' */				      \
      REF (form_unknown),	/* for '+' */				      \
      REF (form_unknown),	/* for '-' */				      \
      ...
    };
    /* Step 3b: after processing first 'l' modifier.  */		      \
    static JUMP_TABLE_TYPE step3b_jumps[30] =				      \
    {									      \
      REF (form_unknown),						      \
      REF (form_unknown),	/* for ' ' */				      \
      REF (form_unknown),	/* for '+' */				      \
      REF (form_unknown),	/* for '-' */				      \
      ...
    }
#define STEP4_TABLE							      \
    /* Step 4: processing format specifier.  */				      \
    static JUMP_TABLE_TYPE step4_jumps[30] =				      \
    {									      \
      REF (form_unknown),						      \
      REF (form_unknown),	/* for ' ' */				      \
      REF (form_unknown),	/* for '+' */				      \
      REF (form_unknown),	/* for '-' */				      \
      ...
    }
```

딱 봐도 매우 노가다인 것처럼 보이는 이 코드는, 실제로 노가다입니다. 서식 문자에서 사용될 수 있는 문자들의 후보들 각각에 대해, 어떤 문자가 나왔을 때 어떤 레이블로 점프할지를 문자 하나 하나 정의해둔 것입니다. 예를 들어 % 뒤 +가 바로 나오는 것은 가능하기 때문에 `step0_jumps` 표에서는 `flag_plus` 레이블로 점프하도록 지정해두지만, 다른 위치에서 나올 수는 없기 때문에 나머지 표에서는 모두 잘못된 서식 문자임을 나타내는 `form_unknown`으로 보내는 것을 볼 수 있습니다. Step 0 ~ 4의 표는 미리 정의된 문자들의 클래스 순서에 따라 만들어져 있는데, 이에 대해서는 잠시 뒤에 설명하겠습니다.

다시 `vfprintf` 함수로 돌아와서, 루프를 돌 때마다 가장 먼저 실행하게 되는 코드는 다음과 같습니다.

```c
/* Get current character in format string.  */
JUMP (*++f, step0_jumps);
```

서식 문자의 첫 글자에 대해 `JUMP`라는 매크로를 호출하는데, 이 매크로는 다음과 같이 정의되어 있습니다.

```c
#define CHAR_CLASS(Ch) (jump_table[(INT_T) (Ch) - L_(' ')])

...

# define JUMP(ChExpr, table)						      \
      do								      \
	{								      \
	  const void *ptr;						      \
	  spec = (ChExpr);						      \
	  ptr = NOT_IN_JUMP_RANGE (spec) ? REF (form_unknown)		      \
	    : table[CHAR_CLASS (spec)];					      \
	  goto *ptr;							      \
	}								      \
      while (0)
```

`JUMP` 매크로는 해당 문자 `ChExpr`을 처리할 레이블을 표 `table`로부터 찾아 그곳으로 goto를 수행하는 코드입니다. 어떻게 goto를 하는데 포인터를 사용할 수 있는지 의아할 수 있는데, 이것은 [GCC 확장 기능 2](http://www.secmem.org/blog/2020/03/17/gcc-extensions-2/)에서 설명한 바 있는 포인터를 사용한 goto에 해당하는 확장 문법입니다.

`CHAR_CLASS` 매크로는 그 문자가 속한 클래스를 얻어내는 기능을 하는데, 이 목록은 `jump_table` 이라는 배열에 정의되어 있습니다. 공백 문자 `' '`부터 시작하여 아스키 문자들의 순서대로 만들어져 있습니다.

```c
/* This table maps a character into a number representing a class.  In
   each step there is a destination label for each class.  */
static const uint8_t jump_table[] =
  {
    /* ' ' */  1,            0,            0, /* '#' */  4,
	       0, /* '%' */ 14,            0, /* '\''*/  6,
	       0,            0, /* '*' */  7, /* '+' */  2,
	       0, /* '-' */  3, /* '.' */  9,            0,
    /* '0' */  5, /* '1' */  8, /* '2' */  8, /* '3' */  8,
    /* '4' */  8, /* '5' */  8, /* '6' */  8, /* '7' */  8,
    /* '8' */  8, /* '9' */  8,            0,            0,
	       0,            0,            0,            0,
	       0, /* 'A' */ 26,            0, /* 'C' */ 25,
	       0, /* 'E' */ 19, /* F */   19, /* 'G' */ 19,
	       0, /* 'I' */ 29,            0,            0,
    /* 'L' */ 12,            0,            0,            0,
	       0,            0,            0, /* 'S' */ 21,
	       0,            0,            0,            0,
    /* 'X' */ 18,            0, /* 'Z' */ 13,            0,
	       0,            0,            0,            0,
	       0, /* 'a' */ 26,            0, /* 'c' */ 20,
    /* 'd' */ 15, /* 'e' */ 19, /* 'f' */ 19, /* 'g' */ 19,
    /* 'h' */ 10, /* 'i' */ 15, /* 'j' */ 28,            0,
    /* 'l' */ 11, /* 'm' */ 24, /* 'n' */ 23, /* 'o' */ 17,
    /* 'p' */ 22, /* 'q' */ 12,            0, /* 's' */ 21,
    /* 't' */ 27, /* 'u' */ 16,            0,            0,
    /* 'x' */ 18,            0, /* 'z' */ 13
  };
```
여기에 쓰인 번호들이 각 문자가 속한 클래스이며, 이 수를 위의 step 0 ~ 4의 테이블의 인덱스로 사용하여 그 클래스에 대한 처리를 담당할 레이블로 점프할 수 있게 됩니다.

## 서식 문자의 처리 과정 ##
각 step은 받아들일 수 있는 문자 클래스가 정해져 있습니다. 예를 들어 '+'와 같은 문자는 오로지 가장 처음인 step 0에만 올 수 있고 이후 step에는 오지 못합니다. 이렇게 서식 문자의 처리를 여러 step으로 나누어 특정 step에서 특정 문자 클래스에 속하는 문자가 나타났을 때 그에 적절한 레이블로 점프시켜 조치하고, 그 뒤의 문자를 다음 step에서 처리하는 것을 반복하는 것이 서식 문자를 처리하는 전체 과정입니다.

정확히는 서식 문자의 특정 파트가 항상 특정 step에서만 다루어지는 것은 아닙니다. 예를 들어 'd'는 정수를 10진수로 출력하기 위한 서식으로 이를 처리하는 레이블이 하나만 존재하지만, 이 'd'라는 문자 앞에 어떤 요소들이 있느냐에 따라 이 레이블로 오는 step은 0부터 4까지 어느 것이든 될 수 있습니다. 여기서는 편의상 각 step이 그 step에서만 처리될 수 있는 요소들을 담당한다고 생각하고 설명하겠습니다. 또한 모든 종류의 서식 문자에 대한 처리 코드를 전부 보기에는 너무 길기 때문에, 각 step에서 담당하는 핵심적인 내용 위주로 서술하겠습니다.

### Step 0 ###
Step 0에서는 서식의 너비를 지정하기 위한 수('1' ~ '9')와 그의 부호에 해당하는 문자(' ', '+', '-')가 오는 경우를 처리합니다. 여기에 해당하는 문자가 발견된 경우에는 step 1으로 넘어가지 않고 다시 한 번 step 0를 다음 문자에 대해 수행해야 하는 경우도 있습니다. 아래는 그 예시입니다.

```c
int space = 0;	/* Use space prefix if no sign is needed.  */
int left = 0;	/* Left-justify output.  */
int showsign = 0;	/* Always begin with plus or minus sign.  */
...
UCHAR_T pad = L_(' ');/* Padding character.  */

...

  /* ' ' flag.  */
LABEL (flag_space):
  space = 1;
  JUMP (*++f, step0_jumps);

  /* '+' flag.  */
LABEL (flag_plus):
  showsign = 1;
  JUMP (*++f, step0_jumps);

  /* The '-' flag.  */
LABEL (flag_minus):
  left = 1;
  pad = L_(' ');
  JUMP (*++f, step0_jumps);
```

`LABEL` 매크로는 이름 그대로 레이블임을 강조하기 위한 매크로로, 실제로 하는 역할은 앞에 `do_`를 붙여 `do_flag_space`와 같은 레이블 전용 이름을 만드는 일입니다.

공백, '+', '-'와 같은 문자들은 아직 서식이 완료된 것은 아니고 특정 옵션을 부여하기 위한 문자들이기 때문에 다음 문자를 계속해서 봐야 합니다. 그래서 이 옵션들이 있었다는 플래그용 변수 값만을 설정해두고, 다시 `JUMP`를 통해 다음 문자가 어떤 클래스에 속하는지 보게 됩니다. 이들의 경우는 아직 너비에 대한 옵션을 전부 본 것이 아니기 때문에 다음 문자는 다시 step 0에서 처리하게 됩니다.

너비값이 정수로 나타나면 비로소 step 1으로 넘어가게 됩니다.

```c
  /* Given width in format string.  */
LABEL (width):
  width = read_int (&f);

  if (__glibc_unlikely (width == -1))
    {
    __set_errno (EOVERFLOW);
    done = -1;
    goto all_done;
    }

  if (*f == L_('$'))
    /* Oh, oh.  The argument comes from a positional parameter.  */
    goto do_positional;
  JUMP (*f, step1_jumps);
```

`width` 레이블은 '%5d'와 같이 서식 문자의 너비가 지정되었을 때 이 중 '1' ~ '9'의 문자가 발견되면 실행하게 되는 레이블입니다. 이 경우 실제 정수값을 `read_int` 함수를 통해 읽고, `step1_jumps` 로 그 뒤의 문자가 정수를 위한 'd'인지, 실수를 위한 'f'인지, 또는 다른 서식인지 등을 확인하게 됩니다.

### Step 1 ###
Step 1은 '%.3f'와 같이 '.' 뒤에 오는 정밀도에 대한 처리를 위한 과정입니다.

```c
LABEL(precision) :
  ++f;
  if (*f == L_('*'))
    {
      const UCHAR_T *tmp;	/* Temporary value.  */

      tmp = ++f;
      if (ISDIGIT(*tmp))
        {
          int pos = read_int(&tmp);

          if (pos == -1)
            {
              __set_errno(EOVERFLOW);
              done = -1;
              goto all_done;
            }

          if (pos && *tmp == L_('$'))
            /* The precision comes from a positional parameter.  */
            goto do_positional;
        }
      prec = va_arg(ap, int);

      /* If the precision is negative the precision is omitted.  */
      if (prec < 0)
        prec = -1;
    }
  else if (ISDIGIT(*f))
    {
      prec = read_int(&f);

      /* The precision was specified in this case as an extremely
         large positive value.  */
      if (prec == -1)
        {
          __set_errno(EOVERFLOW);
          done = -1;
          goto all_done;
        }
    }
  else
    prec = 0;
  JUMP(*f, step2_jumps);
```

`precision` 레이블의 첫 번째 분기는 정밀도가 '*'인 경우, 즉 정밀도 자체가 인자로 주어지는 경우에 대한 처리입니다. 그 다음 문자가 숫자인 경우에는 다시 수를 하나 읽는데, 이 부분은 POSIX 표준에 명시된 positional parameter라고 부르는 것으로 인자의 위치를 직접 지정할 수 있게 해주는 것입니다. 이러한 경우를 처리해주기 위한 `printf_positional`이라는 함수가 있는데, 이에 대한 설명은 이 글에서는 생략하겠습니다. 아무튼, 그 후 주어지는 실제 정밀도는 `va_arg`를 통해 곧바로 인자로부터 읽어와 `prec`에 저장해두게 됩니다.

다음 분기는 정밀도가 숫자로 시작하는 경우, 즉, 직접 정밀도가 주어지는 경우입니다. 이 경우에는 `read_int`로 정밀도의 값을 읽어오는 것으로 해결됩니다. 정밀도가 따로 주어지지 않은 경우는 그냥 0으로 처리됩니다.

이렇게 하면 step 1이 끝나고, step2로 넘어가게 됩니다.

### Step 2 ###
Step 2는 'h'와 'l'을 처리하기 위한 스텝입니다. 'h'는 short를 출력하기 위한 '%hd' 등에 쓰이며, 'l'은 long을 출력하기 위한 '%ld' 등에 사용됩니다. 'h'인지 'l'인지에 따라 다음과 같은 코드들이 실행됩니다.

```c
  /* Process 'h' modifier.  There might another 'h' following.  */
LABEL (mod_half):
  is_short = 1;
  JUMP (*++f, step3a_jumps);

...

  /* Process 'l' modifier.  There might another 'l' following.  */
LABEL (mod_long):
  is_long = 1;
  JUMP (*++f, step3b_jumps);
```

둘 다 step 3로 넘어가지만 정확히는 'h'는 step 3a로 넘어가고 'l'은 step 3b로 넘어갑니다. 둘의 차이는 다음과 같습니다.

### Step 3 ###
'h'와 'l'이 쓰이는 경우는 short와 long만이 아닙니다. char를 출력하기 위한 '%hhd'와 long long을 출력하기 위한 '%lld'처럼 이들을 두 번씩 쓰는 경우가 있습니다. 이들 각각을 처리하기 위한 스텝이 각각 step 3a와 step 3b입니다.

```c
  /* Process 'hh' modifier.  */
LABEL (mod_halfhalf):
  is_short = 0;
  is_char = 1;
  JUMP (*++f, step4_jumps);

...

  /* Process 'L', 'q', or 'll' modifier.  No other modifier is
allowed to follow.  */
LABEL (mod_longlong):
  is_long_double = 1;
  is_long = 1;
  JUMP (*++f, step4_jumps);
```
'hh'의 경우 step 2에서 지정했던 `is_short`를 다시 0으로 바꾸고 `is_char`를 1로 만들어주어 이 서식이 char형을 위한 것임을 표시해줍니다. 'll'의 경우 대신 'q'나 'L'로도 여기에 들어올 수 있으며, 아직 이것이 long long인지 long double인지는 모르기 때문에 일단 두 플래그를 모두 설정해두게 됩니다. `is_longlong`이라는 플래그도 있는데, 이는 정확히는 매크로로 선언되어 long과 long long의 크기가 다른 경우 `is_long_double`의 설정을 따라가게 하고, 같은 경우에는 그냥 long을 처리하는 루트를 따라가도록 0으로 설정됩니다.

### Step 4 ###
모든 설정이 끝나면 마지막으로 어떤 자료형에 대해 어떤 양식으로 출력할지를 결정하는 형식 지정자가 나옵니다. 'd', 'f', 'c', 's'와 같은 것들이 해당되는데, 결국 모든 서식은 이에 해당하는 것들 중 하나로 끝나게 됩니다. 일부 서식의 경우는 step 1 ~ 3이 존재할 수 없어 더 이전 단계에서 바로 점프되는 것만 가능한 경우도 있습니다.

이 단계를 처리하는 레이블들은 다음과 같은 무한 루프 속에 들어있습니다.

```c
/* Process current format.  */
while (1)
  {
    process_arg (((struct printf_spec *) NULL));
    process_string_arg (((struct printf_spec *) NULL));

    LABEL (form_unknown):
    if (spec == L_('\0'))
      {
        /* The format string ended before the specifier is complete.  */
        __set_errno (EINVAL);
        done = -1;
        goto all_done;  
      }

    /* If we are in the fast loop force entering the complicated
       one.  */
    goto do_positional;
  }

/* The format is correctly handled.  */
++nspecs_done;
```

문자열이 아닌 서식을 처리하는 레이블들은 `process_arg` 매크로 내에, 문자열 서식을 처리하는 레이블들은 `process_string_arg` 매크로 내에 있습니다. `form_unknown`은 잘못된 서식을 예외처리하기 위한 레이블입니다.

`process_arg` 매크로는 다음과 같이 정의되어 있습니다.

```c
#define process_arg(fspec)						      \
      /* Start real work.  We know about all flags and modifiers and	      \
	 now process the wanted format specifier.  */			      \
    LABEL (form_percent):						      \
      /* Write a literal "%".  */					      \
      outchar (L_('%'));						      \
      break;								      \
									      \
    LABEL (form_integer):						      \
      /* Signed decimal integer.  */					      \
      ...
    LABEL (form_unsigned):						      \
      /* Unsigned decimal integer.  */					      \
      ...
    ...
```

이전 단계까지 모든 설정과 플래그들을 모두 세팅해 두었으므로 여기서는 그 설정에 맞추어 실제로 출력하는 일을 담당합니다. `va_arg`를 통해 다음 인자의 값을 얻어오고, 그 인자들을 적절한 문자열 형식으로 만들어준 뒤 `outchar`와 `outstring` 매크로를 이용해 출력합니다. 이 매크로들은 아래와 같이 만들어져 있습니다.

```c
# define PUT(F, S, N)	_IO_sputn ((F), (S), (N))
# define PUTC(C, F)	_IO_putc_unlocked (C, F)

...

#define	outchar(Ch)							      \
  do									      \
    {									      \
      const INT_T outc = (Ch);						      \
      if (PUTC (outc, s) == EOF || done == INT_MAX)			      \
	{								      \
	  done = -1;							      \
	  goto all_done;						      \
	}								      \
      ++done;								      \
    }									      \
  while (0)

static inline int
outstring_func (FILE *s, const UCHAR_T *string, size_t length, int done)
{
  assert ((size_t) done <= (size_t) INT_MAX);
  if ((size_t) PUT (s, string, length) != (size_t) (length))
    return -1;
  return done_add_func (length, done);
}

#define outstring(String, Len)						\
  do									\
    {									\
      const void *string_ = (String);					\
      done = outstring_func (s, string_, (Len), done);			\
      if (done < 0)							\
	goto all_done;							\
    }									\
   while (0)
```

`PUT`와 `PUTC`는 실제로 출력 파일에 출력하는 함수를 호출해주는 매크로입니다. `outchar`는 한 글자를 출력하고, `outstring`은 문자열을 출력하는데, 둘 다 출력 도중 문제(EOF 혹은 한 번에 출력할 수 있는 최대 개수 초과)가 발생한 경우에는 `all_done` 레이블로 건너뛰는 부분을 포함하고 있습니다.

이렇게 모든 단계를 완료하고 나면 하나의 서식 문자에 대한 처리가 끝나게 됩니다. 이제 다음 서식 문자를 찾고, 그 서식 문자 전까지의 일반 문자열 부분을 출력한 뒤, 루프를 돌아 다음 서식 문자에 대한 처리를 수행하는 것을 반복하면 됩니다.

## 마치며 ##
지금까지 살펴본 `vfprintf` 함수의 내용은 glibc가 어떤 형태로 만들어져 있는지 그 전형적인 모습을 보여줍니다. 많은 노가다의 흔적이 보일 뿐 아니라, 수많은 매크로들을 적극 활용하여 효율성을 높이고, 심지어는 거대한 코드 덩어리를 통째로 매크로로 만들거나 수십 개의 레이블과 goto문까지도 최적화에 이용하는 모습을 보여주고 있습니다. 때로는 매크로 내의 코드가 특정 함수에만 있는 지역 변수를 그대로 사용하는 등 상식을 벗어난 기법들까지 동원하고 있습니다. 겉보기에는 간단해 보일 것 같은 `printf` 함수 하나에도, 이렇게 안을 들여다 보면 많은 최적화의 노력이 얽혀 복잡하지만 효율적인 코드를 만들어내고 있는 것을 볼 수 있습니다.

그런데 아직까지 살펴본 것은 전체적인 실행 흐름에 불과합니다. 각 서식 문자를 처리하는 구체적인 부분은 보지도 않았습니다. 이 부분에 대한 분석은 다음을 기약하도록 하겠습니다.
